# adapted from https://github.com/NinaWie/NeurIPS2021-traffic4cast

from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

# from model.configs import configs


class PatchUncertainty:
    # def __init__(self, model_path, static_map_arr=None, device="cuda",
    #              radius=50, stride=30, model_str="unet_patches", **kwargs):
    def __init__(self, radius: int, stride: int, static_map_arr = None):
        self.radius = radius
        self.stride = stride
        self.static_map = static_map_arr
        # self.device = device
        # self.model_str = model_str
        # self.pre_transform = configs[model_str][]
        # self.model = self.load_model(model_path,
        #                              use_static_map=(self.static_map is not None))
        # self.model = self.model.to(device)
        # self.model.eval()

    def set_pre_transform(self, pre_transform):
        self.pre_transform = pre_transform

    # def load_model(self, path, use_static_map=False):
    #     model_class = configs[self.model_str]["model_class"]
    #     model_config = configs[self.model_str].get("model_config", {})

    #     if use_static_map:
    #         model_config["in_channels"] += 9

    #     model = model_class(**model_config, img_len=2 * self.radius)
    #     loaded_dict = torch.load(path, map_location=torch.device("cpu"))
    #     model.load_state_dict(loaded_dict["model"])

    #     return model

    def get_input_patches(self, one_hour):
        patch_collection, avg_arr, index_arr, data_static = self.create_patches(
            one_hour, self.radius, self.stride, self.static_map)
        # print("Number of patches per cell", np.mean(avg_arr), np.median(avg_arr))

        inp_patch = self.pre_transform(patch_collection)
        if self.static_map is not None:
            data_static = self.pre_transform(np.expand_dims(data_static, axis=-1))
            inp_patch = torch.cat((inp_patch, data_static), dim=1)

        return inp_patch, avg_arr, index_arr

    def create_patches(self, one_hour, radius, stride, static_map):
    
        tlen, xlen, ylen, chlen = one_hour.shape
        # print(xlen, (xlen - 2*radius), stride, (xlen - 2*radius) // stride)
        nr_in_x = (xlen - 2 * radius) // stride + 2
        nr_in_y = (ylen - 2 * radius) // stride + 2
    
        patch_collection = np.zeros((nr_in_x * nr_in_y, tlen,
                                     radius * 2, radius * 2, chlen))  # to do: 2 auf 8 etc
        avg_arr = np.zeros((xlen, ylen))
        index_arr = np.zeros((nr_in_x * nr_in_y, 4))
    
        if static_map is not None:
            data_static = np.zeros((nr_in_x * nr_in_y, 9, radius * 2, radius * 2))
        else:
            data_static = None

        counter = 0
        for x in range(nr_in_x):
            # special case: we include the end patch
            if x == nr_in_x - 1:
                start_x = xlen - 2 * radius
                end_x = xlen
            else:
                start_x = x * stride
                end_x = x * stride + 2 * radius
    
            for y in range(nr_in_y):
                # special case: we include the end patch
                if y == nr_in_y - 1:
                    start_y = ylen - 2 * radius
                    end_y = ylen
                else:
                    start_y = y * stride
                    end_y = y * stride + 2 * radius
    
                patch_collection[counter] = one_hour[:, start_x:end_x, start_y:end_y, :]

                if static_map is not None:
                    data_static[counter] = static_map[:, start_x:end_x, start_y:end_y]
    
                # remember how often each value was updated
                avg_arr[start_x:end_x, start_y:end_y] += 1
                index_arr[counter] = [start_x, end_x, start_y, end_y]
                counter += 1

        return patch_collection, avg_arr, index_arr, data_static

    # def process_patches(self, inp_patch, avg_arr, index_arr):
    #     out_patch = self.predict_patches(inp_patch)

    #     return out_patch, avg_arr, index_arr

    def predict_patches(self, inp_patch, model, device, internal_batch_size = 50):
        
        """Pass all patches through the model in batches"""

        n_samples = inp_patch.shape[0]
        img_len = inp_patch.shape[2]
        out = torch.zeros(n_samples, 48, img_len, img_len)
        e_b = 0

        for j in range(n_samples // internal_batch_size):
            s_b = j * internal_batch_size
            e_b = (j + 1) * internal_batch_size
            batch_patch = inp_patch[s_b:e_b].to(device)
            out[s_b:e_b] = model(batch_patch).cpu()

        if n_samples % internal_batch_size != 0:
            last_batch = inp_patch[e_b:].to(device)
            out[e_b:] = model(last_batch).cpu()

        return out

    def stitch_patches(self, out_patch, avg_arr, index_arr):
        xlen, ylen = avg_arr.shape
        n_samp = out_patch.shape[0]

        prediction = np.zeros((6, xlen, ylen, 8))
        assert n_samp == index_arr.shape[0]

        for i in range(n_samp):
            (start_x, end_x, start_y, end_y) = tuple(index_arr[i].astype(int).tolist())
            prediction[:, start_x:end_x, start_y:end_y] += out_patch[i]
    
        expand_avg_arr = np.tile(np.expand_dims(avg_arr, 2), 8)
        # avg_prediction = prediction / expand_avg_arr
    
        return prediction / expand_avg_arr

    def get_std(self, out_patch, means, avg_arr, index_arr):
        xlen, ylen = avg_arr.shape
        n_samp = out_patch.shape[0]
    
        std_prediction = np.zeros((6, xlen, ylen, 8))
        assert n_samp == index_arr.shape[0]
    
        for i in range(n_samp):
            (start_x, end_x, start_y, end_y) = tuple(index_arr[i].astype(int).tolist())
            std_prediction[:, start_x:end_x, start_y:end_y] += (out_patch[i] - means[:, start_x:end_x, start_y:end_y]) ** 2
    
        expand_avg_arr = np.tile(np.expand_dims(avg_arr, 2), 8)
        # avg_prediction = np.sqrt(std_prediction / expand_avg_arr)
    
        return np.sqrt(std_prediction / expand_avg_arr)

    @torch.no_grad()
    def __call__(self, device, loss_fct, dataloader, model, samp_limit,
                 parallel_use, post_transform) -> Tuple[torch.Tensor, float]:

        model.eval()
        loss_sum = 0
        bsize = dataloader.batch_size
        batch_limit = samp_limit // bsize
        pred = torch.empty( # Pred contains y_true + point pred + uncertainty: (samples, 3, H, W, Ch)
            size=(batch_limit * bsize, 3, 495, 436, 8), dtype=torch.float32, device=device)

        with tqdm(dataloader) as tloader:
            for batch, (X, y) in enumerate(tloader):
                if batch == batch_limit:
                    break

                # X: tensor (1, 12, 495, 436, 8), y: tensor (1, 6, 495, 436, 8)

                # inp_patch: tensor (patch_x*patch_y, 12*Ch, 2*radius+pad, 2*radius+pad) e.g. (15*13, 12*8, 112, 112)
                # avg_arr: np.array (495, 436), index_arr: np.array (15*13, 4)
                inp_patch, avg_arr, index_arr = self.get_input_patches(X.squeeze(dim=0))
                # print(inp_patch.shape, avg_arr.shape, index_arr.shape)

                # out: tensor (15*13, 6*Ch, 112, 112)
                out = self.predict_patches(inp_patch, model, device)

                # out_patch: np.array (15*13, 6, 100, 100, 8)
                out_patch = post_transform(out).numpy()
                del inp_patch, out

                # Prediction: average over patches, np.array (6, H, W, Ch)
                y_pred = self.stitch_patches(out_patch, avg_arr, index_arr)
                # Uncertainty: std over patches, np.array (6, H, W, Ch)
                std_pred = self.get_std(out_patch, y_pred, avg_arr, index_arr)

                loss = loss_fct(torch.from_numpy(y_pred), y.squeeze(dim=0))
                # y_pred: tensor (1, 3, H, W, Ch), only consider pred horizon 1h
                y_pred = torch.stack((y.squeeze(dim=0),
                                      torch.from_numpy(y_pred).to(dtype=torch.float),
                                      torch.from_numpy(std_pred).to(dtype=torch.float).clamp(min=1e-4)
                                      ), dim=0)[:, 5, ...].clamp(0, 255).unsqueeze(dim=0)

                loss_sum += float(loss.item())
                loss_test = float(loss_sum/(batch+1))
                tloader.set_description(f"Batch {batch+1}/{batch_limit} > eval")
                tloader.set_postfix(loss = loss_test)
          
                assert pred[(batch * bsize):(batch * bsize + bsize)].shape == y_pred.shape
                pred[(batch * bsize):(batch * bsize + bsize)] = y_pred # Fill slice
                del X, y, y_pred, std_pred

        return pred, loss_test


# =============================================================================
# def std_v1(out_patch, index_arr, out_shape):
#     std_preds = np.zeros(out_shape)
# 
#     for p_x in range(495):
#         for p_y in range(436):
#             pixel = (p_x, p_y)
#             # find the patches corresponding to this pixel and get the
#             # middleness and pred per patch
#             preds = []
#             for j, inds in enumerate(index_arr):
#                 x_s, x_e, y_s, y_e = inds
#                 if x_s <= pixel[0] and x_e > pixel[0] and y_s <= pixel[1] and y_e > pixel[1]:
#                     rel_x, rel_y = int(pixel[0] - x_s), int(pixel[1] - y_s)
#                     # what values were predicted for this pixel?
#                     pred_pixel = out_patch[j, :, rel_x, rel_y, :]
#                     # how much in the middle is a pixel?
#                     preds.append(pred_pixel)
#             std_preds[:, pixel[0], pixel[1], :] = np.std(preds, axis=0)
#     return std_preds
# =============================================================================
