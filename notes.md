## Macbook Runs

### Batch Size and dType

| dType | Batch Size | T    | DT (msec)   | Tokens / Sec |
| ----- | ---------- | ---- | ----------- | ------------ |
| FP32  | 16         | 1024 | Error       | Error        |
| FP32  | 8          | 1024 | 10000-28000 | 700-300      |
| FP32  | 4          | 1024 | 990         | 4120         |
| FP32  | 2          | 1024 | 515         | 3980         |
| FP32  | 1          | 1024 | 276         | 3700         |
| **BF16**  | **4**          | **1024** | **950**         | **4320**         |
| BF16  | 2          | 1024 | 500         | 4100         |
| BF16  | 1          | 1024 | 272         | 3750         |

Highest tok/sec: 4320.  (dtype=FP32, B=4, T=1024)

First row: 

- Errors due to Metal memory limits of 30GB reached.
- Droppped down B to 2 to keep it fast enough and within limits.

Second row:

- Too few tokens per sec and takes too long. Looks like it bottlenecks at 8 batch size, maybe due to HBM utilization.

# Compile

| Compile | dType    | Batch Size | T        | DT (msec) | Tokens / Sec |
| ------- | -------- | ---------- | -------- | --------- | ------------ |
| False   | **BF16** | 4          | 1024     | 950       | 4320         |
| True    | **BF16** | **4**      | **1024** | 750       | **5480**     |

## Flash Attention

| Flash Attention | Compile | dType      | Batch Size | T   | DT (msec) | Tokens / Sec |
| --------------- | ------- | ---------- | --------   | --- |     ---   | ------------ |
| False           | True    | BF16       | 4   | 1024      | 750|5460         |
| True            | True    | BF16       | 4   | 1024      |770 | 5340         |

Surprisingly, flash attention runs slower.

## Pretty numbers (vocab. size 53012)

| Vocab Size | Flash Attention | Compile | dType | Batch Size | T    | DT (msec) | Tokens / Sec |
| ---------- | --------------- | ------- | ----- | ---------- | ---- | --------- | ------------ |
| 50257      | False           | True    | BF16  | 4          | 1024 | 750       | 5460         |
| 53012      | True            | True    | BF16  | 4          | 1024 | 750       | 5460         |

Increasing vocab size to a number that's multiple of 2,4,8,16 enables better use of matrix kernels.