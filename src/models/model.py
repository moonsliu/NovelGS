import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pytorch_lightning as pl
from diffusers.schedulers import DDIMScheduler
from einops import rearrange
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import make_grid, save_image

from src.utils.train_util import instantiate_from_config


class DMV3D(pl.LightningModule):
    def __init__(
        self,
        gs_decoder_config,
        gs_renderer_config,
        scheduler_config,
        image_size=256,
        drop_cond_prob=0.1,
        lpips_ratio=0.0,
        n_clean_views = 4,
        n_noisy_views = 4,
        num_inference_steps = 50,
        unconditional_guidance_scale = 5.,
        guidance_rescale = 0.7,
        eta=1,
        load_grm=False,
        rescale_noise_cfg=False,
        freeze_model = False,
        clean_view_only = False,
        downsample_lpips = True,
        grm_checkpoint = None,
        image_log_every_n_steps = 500,
    ):
        super(DMV3D, self).__init__()

        self.image_size = image_size
        self.drop_cond_prob = drop_cond_prob
        self.lpips_ratio = lpips_ratio
        self.num_inference_steps = num_inference_steps
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.guidance_rescale = guidance_rescale
        self.rescale_noise_cfg = rescale_noise_cfg
        self.eta = eta
        self.clean_view_only = clean_view_only
        self.downsample_lpips = downsample_lpips
        self.grm_checkpoint = grm_checkpoint
        self.image_log_every_n_steps = image_log_every_n_steps

        self.n_clean_views = n_clean_views
        self.n_noisy_views = n_noisy_views

        self.scheduler = DDIMScheduler(**scheduler_config)

        # init modules
        self.gs_decoder = instantiate_from_config(gs_decoder_config)
        self.load_grm = load_grm
        self.freeze_model = freeze_model
        if load_grm:
            if not grm_checkpoint:
                raise ValueError("load_grm=True requires params.grm_checkpoint")
            pretrained_weights = torch.load(grm_checkpoint, map_location=torch.device('cpu'))

            pretrained_weights = {k.replace('visual_encoder.', ''): v for k, v in pretrained_weights.items()}

            unmatched_keys = [k for k, v in self.gs_decoder.state_dict().items() if k in pretrained_weights and v.size() != pretrained_weights[k].size()]

            self.gs_decoder.load_state_dict(pretrained_weights, strict=False)

            self.params_to_update = []
            self.params_with_higher_lr = []

            for name, param in self.gs_decoder.named_parameters():
                if name in unmatched_keys:
                    self.params_with_higher_lr.append(param)
                else:
                    self.params_to_update.append(param)
        elif freeze_model:
            if not grm_checkpoint:
                raise ValueError("freeze_model=True requires params.grm_checkpoint")
            pretrained_weights = torch.load(grm_checkpoint, map_location=torch.device('cpu'))
            pretrained_weights = {k.replace('visual_encoder.', ''): v for k, v in pretrained_weights.items()}
            unmatched_keys = [k for k, v in self.gs_decoder.state_dict().items() if k in pretrained_weights and v.size() != pretrained_weights[k].size()]

            # 分组参数：形状不匹配的参数和其他参数
            self.params_to_update = []
            self.params_with_higher_lr = []

            # 找出当前模型中与预训练权重形状不匹配的键
            for name, param in self.gs_decoder.named_parameters():
                if name in unmatched_keys:
                    self.params_with_higher_lr.append(param)  # 形状不匹配的参数
                else:
                    self.params_to_update.append(param)  # 其他参数

        self.gs = instantiate_from_config(gs_renderer_config)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        for param in self.lpips.parameters():
            param.requires_grad_(False)

        # validation output buffer
        self.val_outputs = []

    def prepare_batch_data(self, batch, n_clean_views=1, n_noisy_views=3):
        imgs = batch['imgs_in']    # (B, N, C, H, W)
        c2ws = batch['c2ws']    # (B, N, 4, 4)
        Ks = batch['Ks']        # (B, N, 3, 3)
        alphas = batch['masks']

        clean_imgs = imgs[:, 0:n_clean_views].to(self.device)
        clean_c2ws = c2ws[:, 0:n_clean_views].to(self.device)
        clean_Ks = Ks[:, 0:n_clean_views].to(self.device)
        clean_alphas = alphas[:, 0:n_clean_views].to(self.device)

        noisy_imgs = imgs[:, n_clean_views:n_clean_views+n_noisy_views].to(self.device)
        noisy_c2ws = c2ws[:, n_clean_views:n_clean_views+n_noisy_views].to(self.device)
        noisy_Ks = Ks[:, n_clean_views:n_clean_views+n_noisy_views].to(self.device)
        noisy_alphas = alphas[:, n_clean_views:n_clean_views+n_noisy_views].to(self.device)

        target_imgs = imgs[:, n_clean_views+n_noisy_views:].to(self.device)
        target_c2ws = c2ws[:, n_clean_views+n_noisy_views:].to(self.device)
        target_Ks = Ks[:, n_clean_views+n_noisy_views:].to(self.device)
        target_alphas = alphas[:, n_clean_views+n_noisy_views:].to(self.device)

        return  clean_imgs, clean_c2ws, clean_Ks, clean_alphas, \
                noisy_imgs, noisy_c2ws, noisy_Ks, noisy_alphas, \
                target_imgs, target_c2ws, target_Ks, target_alphas

    # normalization functions
    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1

    def unnormalize_to_zero_to_one(self, t):
        return (t + 1) * 0.5

    # def save_gaussian(self, latent, gs_path, model, opacity_thr=None):
    #     xyz = latent['xyz'][0]
    #     features = latent['feature'][0]
    #     opacity = latent['opacity'][0]
    #     scaling = latent['scaling'][0]
    #     rotation = latent['rotation'][0]

    #     if opacity_thr is not None:
    #         index = torch.nonzero(opacity.sigmoid() > opacity_thr)[:, 0]
    #         xyz = xyz[index]
    #         features = features[index]
    #         opacity = opacity[index]
    #         scaling = scaling[index]
    #         rotation = rotation[index]

    #     pc = model.gaussian_model.set_data(xyz.to(torch.float32), features.to(torch.float32), scaling.to(torch.float32), rotation.to(torch.float32), opacity.to(torch.float32))
    #     pc.save_ply(gs_path)

    def training_step(self, batch, batch_idx):
        # get input
        # clean_imgs, clean_c2ws, clean_Ks, clean_alphas, \
        # noisy_imgs, noisy_c2ws, noisy_Ks, noisy_alphas, \
        # target_imgs, target_c2ws, target_Ks, target_alphas \
        #     = self.prepare_batch_data(batch, self.n_clean_views, self.n_noisy_views)
        clean_imgs, clean_c2ws, clean_Ks, clean_alphas, \
        noisy_imgs, noisy_c2ws, noisy_Ks, noisy_alphas, \
        _, _, _, _ \
            = self.prepare_batch_data(batch, self.n_clean_views, self.n_noisy_views)

        B = noisy_imgs.shape[0]
        timesteps = torch.randint(0, self.scheduler.config['num_train_timesteps'], size=(B,)).to(self.device) # timesteps = torch.tensor([100, 100]).to(self.device)
        noise = torch.randn_like(noisy_imgs)  # timesteps = torch.tensor([0, 0]).to(self.device)
        noisy_imgs = self.scheduler.add_noise(noisy_imgs, noise, timesteps)

        # classifier-free guidance
        drop_mask = torch.rand(B).to(clean_imgs) <= self.drop_cond_prob
        drop_mask = drop_mask.view(B, 1, 1, 1, 1).to(clean_imgs)
        clean_imgs = (1.0 - drop_mask) * clean_imgs

        final_input = torch.cat((clean_imgs, noisy_imgs), dim=1) # (6,5,3,512,512)
        fxfycxcy = torch.cat((clean_Ks, noisy_Ks), dim=1)
        c2ws = torch.cat((clean_c2ws, noisy_c2ws), dim=1)
        camera_feature =  torch.cat([c2ws.flatten(-2, -1), fxfycxcy], -1)

        gs, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=timesteps)

        # filter_mask = torch.nonzero((gs['xyz'].abs() < 1).sum(dim=-1) == 3)
        # for key in gs:
        #     if key == 'depth': continue
        #     if gs[key] is not None:
        #         gs[key] = gs[key][filter_mask[:, 0], filter_mask[:, 1]].unsqueeze(0)
        # self.save_gaussian(gs, './why.ply', self.gs)
        if self.clean_view_only:
            num_points = gs['xyz'].shape[1]
            clean_points = int(num_points / (self.n_clean_views + self.n_noisy_views) * self.n_clean_views)
            gs = {k: v[:, :clean_points] for k, v in gs.items() if k != 'depth'}
        output = self.gs.render(latent=gs,
                output_c2ws=batch['c2ws'],
                output_fxfycxcy=batch['Ks'], bg_color=batch['bg'])

        # bug fixed predict x_0
        rendered_imgs, rendered_alphas = self.normalize_to_neg_one_to_one(output['image']), output['alpha']

        # imgs_in_lpips = self.normalize_to_neg_one_to_one(self.unnormalize_to_zero_to_one(batch["imgs_in"]) * batch["masks"] + torch.ones_like(batch["imgs_in"]) * (1 - batch["masks"]))
        # rendered_imgs_lpips = self.normalize_to_neg_one_to_one(output['image'] * output['alpha'] + torch.ones_like(output['image']) * (1 - output['alpha']))

        # compute losses
        loss, loss_dict = self.compute_loss(rendered_imgs, batch["imgs_out"], rendered_alphas, batch["masks"])

        # logging
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.image_log_every_n_steps > 0 and self.global_step % self.image_log_every_n_steps == 0:
            rendered_imgs = rearrange(rendered_imgs, 'b n c h w -> b c h (n w)')
            rendered_alphas = rearrange(rendered_alphas, 'b n c h w -> b c h (n w)')
            batch["imgs_out"] = rearrange(batch["imgs_out"], 'b n c h w -> b c h (n w)')
            # batch["imgs_in"] = rearrange(batch["imgs_in"], 'b n c h w -> b c h (n w)')
            batch["masks"] = rearrange(batch["masks"], 'b n c h w -> b c h (n w)')

            rendered_imgs = make_grid(rendered_imgs, nrow=1, normalize=True, value_range=(-1, 1))
            rendered_alphas = make_grid(rendered_alphas, nrow=1, normalize=True, value_range=(-1, 1))
            batch["imgs_out"] = make_grid(batch["imgs_out"], nrow=1, normalize=True, value_range=(-1, 1))
            # batch["imgs_in"] = make_grid(batch["imgs_in"], nrow=1, normalize=True, value_range=(-1, 1))
            batch["masks"] = make_grid(batch["masks"], nrow=1, normalize=True, value_range=(-1, 1))

            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            save_image(
                rendered_imgs,
                os.path.join(self.logdir, 'images', f'train_{self.global_step:08d}_{self.global_rank:03d}_render.png')
            )
            save_image(
                rendered_alphas,
                os.path.join(self.logdir, 'images', f'train_{self.global_step:08d}_{self.global_rank:03d}_render_mask.png')
            )
            save_image(
                batch["imgs_out"],
                os.path.join(self.logdir, 'images', f'train_{self.global_step:08d}_{self.global_rank:03d}_gt.png')
            )
            save_image(
                batch["masks"],
                os.path.join(self.logdir, 'images', f'train_{self.global_step:08d}_{self.global_rank:03d}_gt_mask.png')
            )

        return loss

    def compute_loss(self, imgs_pred, imgs_gt, alphas_pred, alphas_gt):
        imgs_pred = rearrange(imgs_pred, 'b n ... -> (b n) ...')
        imgs_gt = rearrange(imgs_gt, 'b n ... -> (b n) ...')
        if imgs_pred.shape[-2:] != imgs_gt.shape[-2:]:
            imgs_gt = F.interpolate(imgs_gt, imgs_pred.shape[-2:], mode='bilinear', align_corners=True)

        loss_mse = F.mse_loss(imgs_pred, imgs_gt)

        # for lpips loss (-1, 1)
        if self.lpips_ratio > 0.0 and self.downsample_lpips:
            loss_lpips = self.lpips(F.interpolate(imgs_pred, size=(imgs_gt.shape[-1]//2, imgs_gt.shape[-1]//2), mode='bilinear', align_corners=True), F.interpolate(imgs_gt, size=(imgs_gt.shape[-1]//2,imgs_gt.shape[-1]//2), mode='bilinear', align_corners=True)) * self.lpips_ratio
        elif self.lpips_ratio > 0.0 and not self.downsample_lpips:
            loss_lpips = self.lpips(F.interpolate(imgs_pred, size=(imgs_gt.shape[-1], imgs_gt.shape[-1]), mode='bilinear', align_corners=True), F.interpolate(imgs_gt, size=(imgs_gt.shape[-1],imgs_gt.shape[-1]), mode='bilinear', align_corners=True)) * self.lpips_ratio
        else:
            loss_lpips = 0.0
        alphas_pred = rearrange(alphas_pred, 'b n ... -> (b n) ...')
        alphas_gt = rearrange(alphas_gt, 'b n ... -> (b n) ...')
        if alphas_pred.shape[-2:] != alphas_gt.shape[-2:]:
            alphas_gt = F.interpolate(alphas_gt, alphas_pred.shape[-2:], mode='bilinear', align_corners=True)

        loss_alpha = F.mse_loss(alphas_pred, alphas_gt)

        # loss_lpips = 0.
        loss = loss_mse + loss_lpips + loss_alpha

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_mse': loss_mse})
        loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
        loss_dict.update({f'{prefix}/loss_alphas': loss_alpha})
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     print("on_before_opt enter")
    #     for name, p in self.gs_decoder.named_parameters():
    #         if p.grad is None:
    #             print(name)
    #     print("on_before_opt exit")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        clean_imgs, clean_c2ws, clean_Ks, clean_alphas, \
        noisy_imgs, noisy_c2ws, noisy_Ks, noisy_alphas, \
        target_imgs, target_c2ws, target_Ks, target_alphas \
            = self.prepare_batch_data(batch, self.n_clean_views, self.n_noisy_views)

        fxfycxcy = torch.cat((clean_Ks, noisy_Ks), dim=1)
        c2ws = torch.cat((clean_c2ws, noisy_c2ws), dim=1)
        camera_feature =  torch.cat([c2ws.flatten(-2, -1), fxfycxcy], -1)

        # sampling
        device = noisy_c2ws.device
        B = noisy_c2ws.shape[0]
        shape = noisy_imgs.shape
        print(f'Data shape for DDIM sampling is {shape}, eta {self.eta}')
        samples = torch.randn(shape, device=device)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((clean_imgs, samples), dim=1)

            gs, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)

            if self.clean_view_only:
                num_points = gs['xyz'].shape[1]
                clean_points = int(num_points / (self.n_clean_views + self.n_noisy_views) * self.n_clean_views)
                gs = {k: v[:, :clean_points] for k, v in gs.items() if k != 'depth'}

            pred_x0 = self.gs.render(latent=gs,
                    output_c2ws=noisy_c2ws,
                    output_fxfycxcy=noisy_Ks, bg_color=torch.ones((B, noisy_c2ws.shape[1], 3)))['image']
            pred_x0 = pred_x0.clamp(0, 1)
            pred_original_sample = pred_x0
            if pred_x0.shape[-1] != clean_imgs.shape[-1]:
                pred_x0 = rearrange(pred_x0, 'b n ... -> (b n) ...')
                pred_x0 = F.interpolate(pred_x0, size=(clean_imgs.shape[-1], clean_imgs.shape[-1]), mode='bilinear', align_corners=True)
                pred_x0 = rearrange(pred_x0, '(b n) ... -> b n ...', b=clean_imgs.shape[0])
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)

            # predict x_prev
            DDIMSchedulerOutput = self.scheduler.step(pred_x0, step, samples, eta=self.eta, return_dict=True)
            samples, _ = DDIMSchedulerOutput[0], DDIMSchedulerOutput[1]

        samples = samples.clamp(-1, 1)
        pred_original_sample = self.normalize_to_neg_one_to_one(pred_original_sample)

        output = {'clean_imgs': clean_imgs, 'noisy_imgs': noisy_imgs, 'sample_imgs': samples, 'pred_original_sample': pred_original_sample}
        self.val_outputs.append(output)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        clean_imgs = torch.cat([output['clean_imgs'] for output in self.val_outputs], 0)     # (B, N, C, H, W)
        noisy_imgs = torch.cat([output['noisy_imgs'] for output in self.val_outputs], 0)     # (B, N, C, H, W)
        sample_imgs = torch.cat([output['sample_imgs'] for output in self.val_outputs], 0)   # (B, N, C, H, W)
        pred_original_sample = torch.cat([output['pred_original_sample'] for output in self.val_outputs], 0)   # (B, N, C, H, W)

        clean_imgs = rearrange(clean_imgs, 'b n c h w -> b c h (n w)')
        noisy_imgs = rearrange(noisy_imgs, 'b n c h w -> b c h (n w)')
        sample_imgs = rearrange(sample_imgs, 'b n c h w -> b c h (n w)')
        pred_original_sample = rearrange(pred_original_sample, 'b n c h w -> b c h (n w)')

        clean_imgs = make_grid(clean_imgs, nrow=1, normalize=True, value_range=(-1, 1))
        noisy_imgs = make_grid(noisy_imgs, nrow=1, normalize=True, value_range=(-1, 1))
        sample_imgs = make_grid(sample_imgs, nrow=1, normalize=True, value_range=(-1, 1))
        pred_original_sample = make_grid(pred_original_sample, nrow=1, normalize=True, value_range=(-1, 1))

        os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
        save_image(
            clean_imgs,
            os.path.join(self.logdir, 'images', f'val_{self.global_step:08d}_{self.global_rank:03d}_clean.png')
        )

        save_image(
            pred_original_sample,
            os.path.join(self.logdir, 'images', f'val_{self.global_step:08d}_{self.global_rank:03d}_original_reso_sample.png')
        )

        save_image(
            noisy_imgs,
            os.path.join(self.logdir, 'images', f'val_{self.global_step:08d}_{self.global_rank:03d}_noisy.png')
        )

        # sample images
        save_image(
            sample_imgs,
            os.path.join(self.logdir, 'images', f'val_{self.global_step:08d}_{self.global_rank:03d}_sample.png')
        )

        # clear buffuer
        self.val_outputs = []

    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'setting learning rate to {lr:.4f} ...')

        params = []

        if self.load_grm:
            params.append({"params": self.params_to_update, "lr": lr})
            # params.append({"params": self.params_with_higher_lr, "lr": lr*10})
        elif self.freeze_model:
            params.append({"params": self.params_to_update, "lr": lr})
        else:
            params.append({"params": self.gs_decoder.parameters(), "lr": lr})

        # optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=lr/4)

        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.90, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4000, eta_min=lr/4)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def val(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
        B = c2ws.shape[0]
        if images.shape[1] == 4:
            V = 5
        else:
            V = 3
        shape = (B, V, 3, self.image_size, self.image_size)
        samples = torch.randn(shape, device=c2ws.device)
        camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((images, samples), dim=1)

            gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)

            pred_x0 = self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws[:,images.shape[1]:],
                    output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((1, 1, 3)))['image']
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
            pred_x0_final = pred_x0

            samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]

        samples = samples.clamp(-1, 1)
        frames =  self.gs.render(latent=gs_cond,
                    output_c2ws=output_c2ws,
                    output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((1, 1, 3)))['image']

        return samples, frames

    def val4to9(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
        B = c2ws.shape[0]
        shape = (B, c2ws.shape[1]-images.shape[1], 3, self.image_size, self.image_size)
        samples = torch.randn(shape, device=c2ws.device)
        camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((images, samples), dim=1)
            gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)
            if self.clean_view_only:
                num_points = gs_cond['xyz'].shape[1]
                clean_points = int(num_points / (self.n_clean_views + self.n_noisy_views) * self.n_clean_views)
                gs_cond = {k: v[:, :clean_points] for k, v in gs_cond.items() if k != 'depth'}
            pred_x0 = self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws[:,images.shape[1]:],
                    output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, samples.shape[1], 3)))['image']
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
            pred_x0.clip(-1,1)

            if self.unconditional_guidance_scale != 1.0:
                final_input_uncond = torch.cat((torch.zeros_like(images), samples), dim=1)
                gs_uncond, _  = self.gs_decoder.forward(final_input_uncond, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)
                if self.clean_view_only:
                    num_points = gs_uncond['xyz'].shape[1]
                    clean_points = int(num_points / (self.n_clean_views + self.n_noisy_views) * self.n_clean_views)
                    gs_uncond = {k: v[:, :clean_points] for k, v in gs_uncond.items() if k != 'depth'}
                pred_x0_uncond = self.gs.render(latent=gs_uncond,
                        output_c2ws=c2ws[:,images.shape[1]:],
                        output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, samples.shape[1], 3)))['image']

                pred_x0_uncond = self.normalize_to_neg_one_to_one(pred_x0_uncond)
                pred_x0_uncond = pred_x0_uncond.clip(-1,1)
                pred_x0_final = pred_x0_uncond + self.unconditional_guidance_scale * (pred_x0 - pred_x0_uncond)
            else:
                pred_x0_final = pred_x0

            samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]


        samples = samples.clamp(-1, 1)
        frames =  self.gs.render(latent=gs_cond,
                    output_c2ws=output_c2ws,
                    output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, output_fxfycxcy.shape[1], 3)))['image']

        return samples, frames, gs_cond


    def val_mesh(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
        B = c2ws.shape[0]
        shape = (B, 1, 3, self.image_size, self.image_size)
        samples = torch.randn(shape, device=c2ws.device)
        camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((images, samples), dim=1)
            gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)
            if self.clean_view_only:
                num_points = gs_cond['xyz'].shape[1]
                clean_points = int(num_points / (self.n_clean_views + self.n_noisy_views) * self.n_clean_views)
                gs_cond = {k: v[:, :clean_points] for k, v in gs_cond.items() if k != 'depth'}

            pred_x0 = self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws[:,images.shape[1]:],
                    output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 1, 3)))['image']
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
            pred_x0.clip(-1,1)

            if self.unconditional_guidance_scale != 1.0:
                final_input_uncond = torch.cat((torch.zeros_like(images), samples), dim=1)
                gs_uncond, _  = self.gs_decoder.forward(final_input_uncond, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)
                if self.clean_view_only:
                    num_points = gs_uncond['xyz'].shape[1]
                    clean_points = int(num_points / (self.n_clean_views + self.n_noisy_views) * self.n_clean_views)
                    gs_uncond = {k: v[:, :clean_points] for k, v in gs_uncond.items() if k != 'depth'}
                pred_x0_uncond = self.gs.render(latent=gs_uncond,
                        output_c2ws=c2ws[:,images.shape[1]:],
                        output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 1, 3)))['image']

                pred_x0_uncond = self.normalize_to_neg_one_to_one(pred_x0_uncond)
                pred_x0_uncond = pred_x0_uncond.clip(-1,1)
                pred_x0_final = pred_x0_uncond + self.unconditional_guidance_scale * (pred_x0 - pred_x0_uncond)
            else:
                pred_x0_final = pred_x0

            samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]


        samples = samples.clamp(-1, 1)
        frames =  self.gs.render(latent=gs_cond,
                    output_c2ws=output_c2ws,
                    output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, output_fxfycxcy.shape[1], 3)))['image']

        return samples, frames, gs_cond

    def val4to6(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
        B = c2ws.shape[0]
        shape = (B, 2, 3, self.image_size, self.image_size)
        samples = torch.randn(shape, device=c2ws.device)
        camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((images, samples), dim=1)

            gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)

            pred_x0 = self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws[:,images.shape[1]:],
                    output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 1, 3)))['image']
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
            pred_x0_final = pred_x0

            samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]

        samples = samples.clamp(-1, 1)
        frames =  self.gs.render(latent=gs_cond,
                    output_c2ws=output_c2ws,
                    output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, 1, 3)))['image']

        return samples, frames, gs_cond

    def valnoise(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
        B = c2ws.shape[0]
        shape = (B, 1, 3, self.image_size, self.image_size)
        samples = torch.randn(shape, device=c2ws.device)
        camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((images, samples), dim=1)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)
            # gs_cond = {k: v[:, -262144:] for k, v in gs_cond.items()}
            pred_x0 = self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws[:,images.shape[1]:],
                    output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 1, 3)))['image']
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
            pred_x0_final = pred_x0

            samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]

        samples = samples.clamp(-1, 1)
        frames =  self.gs.render(latent=gs_cond,
                    output_c2ws=output_c2ws,
                    output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, output_c2ws.shape[1], 3)))['image']
        frames_new =  self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws,
                    output_fxfycxcy=fxfycxcy, bg_color=torch.ones((B, c2ws.shape[1], 3)))['image']

        return samples, frames, gs_cond, frames_new

    # def val_all(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
    #     B = c2ws.shape[0]
    #     shape = (B, 1, 3, self.image_size, self.image_size)
    #     samples = torch.randn(shape, device=c2ws.device)
    #     camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

    #     # set step values
    #     self.scheduler.set_timesteps(self.num_inference_steps)
    #     print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

    #     for step in tqdm(self.scheduler.timesteps):
    #         ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

    #         # decode gaussians: conditional
    #         final_input = torch.cat((images, samples), dim=1)

    #         gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)

    #         pred_x0 = self.gs.render(latent=gs_cond,
    #                 output_c2ws=c2ws[:,images.shape[1]:],
    #                 output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 1, 3)))['image']
    #         pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
    #         pred_x0_final = pred_x0

    #         samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]

    #     samples = samples.clamp(-1, 1)
    #     frames =  self.gs.render(latent=gs_cond,
    #                 output_c2ws=output_c2ws,
    #                 output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, 1, 3)))['image']

    #     return samples, frames, gs_cond

    def val_clean(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
        B = c2ws.shape[0]
        shape = (B, 1, 3, self.image_size, self.image_size)
        samples = torch.randn(shape, device=c2ws.device)
        camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((images, samples), dim=1)

            gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)
            gs_cond = {k: v[:, :-262144] for k, v in gs_cond.items()}
            pred_x0 = self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws[:,images.shape[1]:],
                    output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 1, 3)))['image']
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
            pred_x0_final = pred_x0

            samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]

        samples = samples.clamp(-1, 1)
        frames =  self.gs.render(latent=gs_cond,
                    output_c2ws=output_c2ws,
                    output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, output_fxfycxcy.shape[1], 3)))['image']

        return samples, frames, gs_cond

    def val_4to6(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
        B = c2ws.shape[0]
        shape = (B, 2, 3, self.image_size, self.image_size)
        samples = torch.randn(shape, device=c2ws.device)
        camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((images, samples), dim=1)

            gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)
            # gs_cond = {k: v[:, :-262144] for k, v in gs_cond.items()}
            pred_x0 = self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws[:,images.shape[1]:],
                    output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 2, 3)))['image']
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
            pred_x0_final = pred_x0

            samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]

        samples = samples.clamp(-1, 1)
        frames =  self.gs.render(latent=gs_cond,
                    output_c2ws=output_c2ws,
                    output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, output_fxfycxcy.shape[1], 3)))['image']

        return samples, frames, gs_cond

    def val_5to6(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
        B = c2ws.shape[0]
        shape = (B, 1, 3, self.image_size, self.image_size)
        samples = torch.randn(shape, device=c2ws.device)
        camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

        # set step values
        self.scheduler.set_timesteps(self.num_inference_steps)
        print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

        for step in tqdm(self.scheduler.timesteps):
            ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

            # decode gaussians: conditional
            final_input = torch.cat((images, samples), dim=1)

            gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)
            # gs_cond = {k: v[:, :-262144] for k, v in gs_cond.items()}
            pred_x0 = self.gs.render(latent=gs_cond,
                    output_c2ws=c2ws[:,images.shape[1]:],
                    output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 1, 3)))['image']
            pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
            pred_x0_final = pred_x0

            samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]

        samples = samples.clamp(-1, 1)
        frames =  self.gs.render(latent=gs_cond,
                    output_c2ws=output_c2ws,
                    output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, output_fxfycxcy.shape[1], 3)))['image']

        return samples, frames, gs_cond

    # def val_repeat(self, images, c2ws, fxfycxcy, output_c2ws, output_fxfycxcy):
    #     B = c2ws.shape[0]
    #     shape = (B, 1, 3, self.image_size, self.image_size)
    #     samples = torch.randn(shape, device=c2ws.device)
    #     camera_feature =  torch.cat([c2ws.flatten(-2,-1), fxfycxcy], -1)

    #     # set step values
    #     self.scheduler.set_timesteps(self.num_inference_steps)
    #     print(f"Running DDIM Sampling with {self.num_inference_steps} timesteps")

    #     for step in tqdm(self.scheduler.timesteps):
    #         ts = torch.full((B,), step, device=c2ws.device, dtype=torch.long)

    #         # decode gaussians: conditional
    #         final_input = torch.cat((images, samples), dim=1)

    #         gs_cond, _  = self.gs_decoder.forward(final_input, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws, t=ts)

    #         pred_x0 = self.gs.render(latent=gs_cond,
    #                 output_c2ws=c2ws[:,images.shape[1]:],
    #                 output_fxfycxcy=fxfycxcy[:,images.shape[1]:], bg_color=torch.ones((B, 1, 3)))['image']
    #         pred_x0 = self.normalize_to_neg_one_to_one(pred_x0)
    #         pred_x0_final = pred_x0

    #         samples = self.scheduler.step(pred_x0_final, step, samples, eta=self.eta, return_dict=False)[0]

    #     samples = samples.clamp(-1, 1)
    #     frames =  self.gs.render(latent=gs_cond,
    #                 output_c2ws=output_c2ws,
    #                 output_fxfycxcy=output_fxfycxcy, bg_color=torch.ones((B, 1, 3)))['image']

    #     return samples, frames, gs_cond