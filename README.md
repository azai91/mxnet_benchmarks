# Mxnet Optimization

## TLDR

- Remember that nd operations are async, this can screw with your benchmark timings
- When loading your bath

I ran into an issue earlier this month where

```
>>> allocate_gpu_memory_evenly()
('Init on GPU', 0)
('Time to init GPU', 0, 4.7590320110321045)
('Init on GPU', 1)
('Time to init GPU', 1, 0.09308409690856934)
('Init on GPU', 2)
('Time to init GPU', 2, 0.0710139274597168)
('Init on GPU', 3)
('Time to init GPU', 3, 0.06910014152526855)
('Total time to init all four GPUS', 4.992389917373657)
```

```
>>> allocate_gpu_memory_one()
('Init on GPU', 0)
('Time to init GPU', 0, 0.006237030029296875)
('Init on GPU', 1)
('Time to init GPU', 1, 0.10968303680419922)
('Init on GPU', 2)
('Time to init GPU', 2, 0.11631011962890625)
('Init on GPU', 3)
('Time to init GPU', 3, 0.1155390739440918)
('Total time to init all four GPUS', 0.348193883895874)
```

```
>>> allocate_gpu_memory_async()
('Init on GPU', 0)
('Time to init GPU', 0, 8.58306884765625e-05)
('Init on GPU', 1)
('Time to init GPU', 1, 5.3882598876953125e-05)
('Init on GPU', 2)
('Time to init GPU', 2, 5.507469177246094e-05)
('Init on GPU', 3)
('Time to init GPU', 3, 4.601478576660156e-05)
('Total time to init all four GPUS', 0.0003139972686767578)
```

