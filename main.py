import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import *

from nerf.gui import NeRFGUI
from nerf.blender import BlenderDataset
from nerf.llff import LLFFDataset
# from nerf.llff_pre import LLFFDataset
# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--text_bg', default=None, help="background prompt")
    parser.add_argument('--negative', default='', type=str,
                        help="negative text prompt")
    parser.add_argument('-O', action='store_true',
                        help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true',
                        help="equals --backbone vanilla --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_mesh', action='store_true',
                        help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=1,
                        help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion',
                        help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)

    # training options
    parser.add_argument('--iters', type=int, default=10000,
                        help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="max learning rate")
    parser.add_argument('--warm_iters', type=int,
                        default=500, help="training iters")
    parser.add_argument('--min_lr', type=float, default=1e-4,
                        help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true',
                        help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=512,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64,
                        help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32,
                        help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo', action='store_true',
                        help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=1000,
                        help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5,
                        help="likelihood of sampling camera location uniformly on the sphere surface area")
    # model options
    parser.add_argument('--bg_radius', type=float, default=0,
                        help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=10,
                        help="max (center) density for the gaussian density blob")
    parser.add_argument('--blob_radius', type=float, default=0.3,
                        help="control the radius for the gaussian density blob")
    # network backbone
    parser.add_argument('--fp16', action='store_true',
                        help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='grid',
                        choices=['grid', 'vanilla'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan',
                        choices=['adan', 'adam', 'adamw'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.0',
                        choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None,
                        help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=400,
                        help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=400,
                        help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true',
                        help="add jitters to the randomly sampled camera poses")

    # dataset options

    parser.add_argument('--bound', type=float, default=1.3,
                        help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1,
                        help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*',
                        default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*',
                        default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true',
                        help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--suppress_face', action='store_true',
                        help="also use negative dir text prompt.")
    parser.add_argument('--angle_overhead', type=float, default=30,
                        help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60,
                        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    # GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=3,
                        help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60,
                        help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60,
                        help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0,
                        help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1,
                        help="GUI rendering max sample per pixel")

    # for scene
    parser.add_argument('--img_wh', nargs="+", type=int, default=[504, 378],  # [252, 189]
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--data_dir', type=str, default='../')
    parser.add_argument('--exp_name', type=str, default='flower')
    parser.add_argument('--data_type', type=str, default='llff')
    parser.add_argument('--spheric_poses', action='store_true')

    parser.add_argument('--pretrained', type=bool, default=False)

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.dir_text = False
        opt.cuda_ray = True

    elif opt.O2:
        # only use fp16 if not evaluating normals (else lead to NaNs in training...)
        if opt.albedo:
            opt.fp16 = True
        opt.dir_text = False
        opt.backbone = 'vanilla'

    if opt.albedo:
        opt.albedo_iters = opt.iters

    from nerf.network_grid import NeRFNetwork

    print(opt)

    seed_everything(opt.seed)

    model = NeRFNetwork(opt)

    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        guidance = None  # no need to load guidance model at test
        clip_guidance = None
        trainer = Trainer('df', opt, model, guidance, clip_guidance, device=device,
                          workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            # test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

            dargs = {
                'root_dir': os.path.join(opt.data_dir, opt.exp_name),
                'img_wh': tuple(opt.img_wh)}

            print("test data.......")
            if opt.data_type == 'llff':
                dargs['spheric_poses'] = opt.spheric_poses
                dargs['val_num'] = 1
                # print("test data .......llff")
                test_dataset = LLFFDataset(
                    device=device, split='test', **dargs)  # 修改过
            else:
                test_dataset = BlenderDataset(
                    split='test', device=device, **dargs)  # 修改过

            test_loader = DataLoader(test_dataset,  # 修改过
                                     shuffle=False,
                                     num_workers=0,
                                     batch_size=1)
            # print("test loader.......................")
            trainer.test(test_loader, write_video=True)

            if opt.save_mesh:
                trainer.save_mesh(resolution=256)

    else:

        dargs = {
            'root_dir': os.path.join(opt.data_dir, opt.exp_name),
            'img_wh': tuple(opt.img_wh)}

        if opt.data_type == 'llff':
            dargs['spheric_poses'] = opt.spheric_poses
            dargs['val_num'] = 1
            train_dataset = LLFFDataset(
                device=device, split='train', **dargs)
        else:
            train_dataset = BlenderDataset(
                split='train', device=device, **dargs)
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=0,
                                  batch_size=1,
                                  pin_memory=False)

        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR

            def optimizer(model): return Adan(model.get_params(
                5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            def optimizer(model): return torch.optim.Adam(
                model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.backbone == 'vanilla':
            def warm_up_with_cosine_lr(iter): return iter / opt.warm_iters if iter <= opt.warm_iters \
                else max(0.5 * (math.cos((iter - opt.warm_iters) / (opt.iters - opt.warm_iters) * math.pi) + 1),
                         opt.min_lr / opt.lr)

            def scheduler(optimizer): return optim.lr_scheduler.LambdaLR(
                optimizer, warm_up_with_cosine_lr)
        else:
            # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
            def scheduler(optimizer): return optim.lr_scheduler.LambdaLR(
                optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        if opt.pretrained:
            if opt.guidance == 'stable-diffusion':
                from nerf.sd import StableDiffusion
                from nerf.clip import CLIP
                guidance = StableDiffusion(device, opt.sd_version, opt.hf_key)
                clip_guidance = CLIP(device)

            elif opt.guidance == 'clip':
                from nerf.clip import CLIP
                guidance = CLIP(device)
            else:
                raise NotImplementedError(
                    f'--guidance {opt.guidance} is not implemented.')
        else:
            guidance = None
            clip_guidance = None

        trainer = Trainer('df', opt, model, guidance, clip_guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None,
                          fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True, pretrained=opt.pretained)

        if opt.gui:
            trainer.train_loader = train_loader  # attach dataloader to trainer

            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            # valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

            if opt.data_type == 'llff':
                dargs['spheric_poses'] = opt.spheric_poses
                dargs['val_num'] = 1
                val_dataset = LLFFDataset(
                    device=device, split='val', **dargs)  # 修改过
            else:
                val_dataset = BlenderDataset(
                    split='val', device=device, **dargs)  # 修改过
            valid_loader = DataLoader(val_dataset,  # 修改过
                                      shuffle=False,
                                      num_workers=0,
                                      batch_size=1,
                                      pin_memory=False)
            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)
