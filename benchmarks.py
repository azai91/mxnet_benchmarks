import mxnet as mx
import time

gpus = [mx.gpu(i) for i in range(4)]

def warm_up():
    a = mx.nd.random_normal(shape=(299, 300, 30, 60), ctx=gpus[0])
    print('Allocating memory on GPUs')
    for gpu in gpus:
        print('Init on GPU', gpu)
        a.as_in_context(gpu).wait_to_read()

def allocate_gpu_memory_evenly():
    a = mx.nd.random_normal(shape=(299, 300, 30, 60))
    start = time.time()
    for i, gpu in enumerate(gpus):
        print('Init on GPU', i)
        gpu_start = time.time()
        a.as_in_context(gpu).wait_to_read()
        print('Time to init GPU', i, time.time() - gpu_start)
    print('Total time to init all four GPUS', time.time() - start)

def allocate_gpu_memory_one():
    start = time.time()
    a = mx.nd.random_normal(shape=(299, 300, 30, 60), ctx=gpus[0])
    for i, gpu in enumerate(gpus):
        print('Init on GPU', i)
        gpu_start = time.time()
        a.as_in_context(gpu).wait_to_read()
        print('Time to init GPU', i, time.time() - gpu_start)
    print('Total time to init all four GPUS', time.time() - start)


def allocate_gpu_memory_async():
    a = mx.nd.random_normal(shape=(299, 300, 30, 60))
    start = time.time()
    for i, gpu in enumerate(gpus):
        print('Init on GPU', i)
        gpu_start = time.time()
        a.as_in_context(gpu)
        print('Time to init GPU', i, time.time() - gpu_start)
    print('Total time to init all four GPUS', time.time() - start)

