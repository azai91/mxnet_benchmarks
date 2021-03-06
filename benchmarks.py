import mxnet as mx
import time
import numpy as np

gpus = [mx.gpu(i) for i in range(4)]

def warm_up():
    a = mx.nd.random_normal(shape=(299, 300, 30, 60), ctx=gpus[0])
    print('Allocating memory on GPUs')
    for gpu in gpus:
        print('Init on GPU', gpu)
        a.as_in_context(gpu)
    mx.nd.waitall()

# Init array on cpu then move to each GPU
def allocate_gpu_memory_evenly():
    start = time.time()
    a = mx.nd.random_normal(shape=(299, 300, 30, 60), ctx=mx.cpu())
    for i, gpu in enumerate(gpus):
        print('Init on GPU', i)
        gpu_start = time.time()
        a.as_in_context(gpu)
        print('Time to init GPU', i, time.time() - gpu_start)
    mx.nd.waitall()
    print('Total time to init all four GPUS', time.time() - start)

# Init array on one gpu then move to each GPU
def allocate_gpu_memory_one():
    start = time.time()
    a = mx.nd.random_normal(shape=(299, 300, 30, 60), ctx=gpus[0]) # this asychronous
    for i, gpu in enumerate(gpus):
        print('Init on GPU', i)
        gpu_start = time.time()
        a.as_in_context(gpu)
        print('Time to init GPU', i, time.time() - gpu_start)
    mx.nd.waitall()
    print('Total time to init all four GPUS', time.time() - start)

# Init array on one gpu then move to each GPU. Does not wait for copies to complete
def allocate_gpu_memory_async():
    start = time.time()
    a = mx.nd.random_normal(shape=(299, 300, 30, 60))
    for i, gpu in enumerate(gpus):
        print('Init on GPU', i)
        gpu_start = time.time()
        a.as_in_context(gpu)
        print('Time to init GPU', i, time.time() - gpu_start)
    print('Total time to init all four GPUS', time.time() - start)






