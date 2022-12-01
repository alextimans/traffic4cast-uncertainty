# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import shutil
import psutil
import logging
from tabulate import tabulate


def system_status() -> str:

    try:
        import GPUtil
    except ImportError:
        logging.info("GPUtil not installed, returning empty.")
        return " "

    s = "\n" + _make_title("GPU Details")
    try:
        gpus = GPUtil.getGPUs()
        list_gpus = []
        for gpu in gpus:
            # get the GPU id
            gpu_id = gpu.id
            # name of GPU
            gpu_name = gpu.name
            # get % percentage of GPU usage of that GPU
            gpu_load = f"{gpu.load * 100}%"
            # get free memory in MB format
            gpu_free_memory = f"{gpu.memoryFree}MB"
            # get used memory
            gpu_used_memory = f"{gpu.memoryUsed}MB"
            # get total memory
            gpu_total_memory = f"{gpu.memoryTotal}MB"
            # get GPU temperature in Celsius
            gpu_temperature = f"{gpu.temperature} °C"
            gpu_uuid = gpu.uuid

            list_gpus.append((gpu_id, gpu_name, gpu_load, gpu_free_memory,
                              gpu_used_memory, gpu_total_memory, gpu_temperature, gpu_uuid))
        s += tabulate(list_gpus, headers=("id", "name", "load", "free memory",
                                          "used memory", "total memory", "temperature", "uuid"))
    except Exception as e:
        s += str(e)
    s += "\n"

    s += _make_title("System memory usage")
    mem = psutil.virtual_memory()
    virtual_memory_fields = ["total", "available", "percent", "used",
                             "free", "active", "inactive", "buffers",
                             "cached", "shared", "slab"]
    virtual_memory_fields = [f for f in virtual_memory_fields if hasattr(mem, f)]
    s += tabulate([[str(mem.__getattribute__(a)) for a in virtual_memory_fields]], 
                  headers=virtual_memory_fields) 
    s += "\n"

    s += _make_title("Disk usage")
    du = psutil.disk_usage("/")
    du_fields = ["total", "used", "free", "percent"]
    du_fields = [f for f in du_fields if hasattr(du, f)]
    s += tabulate([[str(du.__getattribute__(a)) for a in du_fields]],
                  headers=du_fields)
    s += "\n"

    return s


def _make_title(title):
    s = "\n" + "=" * 40 + " " + title + " " + "=" * 40 + "\n"

    return s


def disk_usage_human_readable(path):
    du = shutil.disk_usage(path)
    div = (1024 * 1024 * 1024)
    s = f"usage(total={du[0]/div:.2f}GB, used={du[1]/div:.2f}GB, free={du[2]/div:.2f}GB)"

    return s
