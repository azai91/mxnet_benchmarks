# Mxnet Optimization

## TLDR

- Remember that nd operations are async, this can screw with your benchmark timings
- When loading your batches into multiple GPUs, you should first asynchronously load the batches onto one GPU then move them over to each GPU

```
>>> allocate_gpu_memory_evenly()
('Init on GPU', 0)
('Time to init GPU', 0, 0.00010704994201660156)
('Init on GPU', 1)
('Time to init GPU', 1, 6.29425048828125e-05)
('Init on GPU', 2)
('Time to init GPU', 2, 5.602836608886719e-05)
('Init on GPU', 3)
('Time to init GPU', 3, 5.1975250244140625e-05)
('Total time to init all four GPUS', 4.655671119689941)
```

```
>>> allocate_gpu_memory_one()
('Init on GPU', 0)
('Time to init GPU', 0, 1.5020370483398438e-05)
('Init on GPU', 1)
('Time to init GPU', 1, 8.988380432128906e-05)
('Init on GPU', 2)
('Time to init GPU', 2, 5.888938903808594e-05)
('Init on GPU', 3)
('Time to init GPU', 3, 5.1975250244140625e-05)
('Total time to init all four GPUS', 0.23697781562805176)
```

```
>>> allocate_gpu_memory_async()
('Init on GPU', 0)
('Time to init GPU', 0, 0.00015401840209960938)
('Init on GPU', 1)
('Time to init GPU', 1, 0.00010204315185546875)
('Init on GPU', 2)
('Time to init GPU', 2, 8.797645568847656e-05)
('Init on GPU', 3)
('Time to init GPU', 3, 8.511543273925781e-05)
('Total time to init all four GPUS', 0.0007338523864746094)
```

