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
| True            | True    | BF16       | 4   | 1024      |770 | 5490         |
| True            | True    | BF16       | 8   | 1024      |770 | 5465   |      

Surprisingly, flash attention runs slower with same batch size, but it enables us to run with batch size 16, with small  gains for token/sec.

## Pretty numbers (vocab. size 50304)

| Vocab Size | Flash Attention | Compile | dType | Batch Size | T    | DT (msec) | Tokens / Sec |
| ---------- | --------------- | ------- | ----- | ---------- | ---- | --------- | ------------ |
| 50257      | False           | True    | BF16  | 4          | 1024 | 750       | 5460         |
| 50304      | False           | True    | BF16  | 4          | 1024 | 745       | 5490         |

Increasing vocab size to a number that's multiple of 128 should enable better use of matrix kernels. Results are slightly better


# CUDA (A100)

## Batch Size
| Vocab Size | Flash Attention | Compile | dType | Batch Size | T    | DT (msec) | Tokens / Sec |
| ---------- | --------------- | ------- | ----- | ---------- | ---- | --------- | ------------ |
| 50257      | False           | True    | BF16  | 4          | 1024 | 55.6        | 73652  |
| 50257      | False           | True    | BF16  | 16          | 1024 | 183.6        | 89267  |
| 50257      | False           | True    | BF16  | 32          | 1024 | 352.8        | 92892  |
| 50257      | False           | True    | BF16  | 64          | 1024 | ERR        | ERR  |

Batch Size 64 is too big for the TPU memory. It requires > 78GB HBM and A100 have 80GB max capacity (79.5GB).


## Flash Attention
| Vocab Size | Flash Attention | Compile | dType | Batch Size  | T    | DT (msec) | Tokens / Sec |
| ---------- | --------------- | ------- | ----- | ----------  | ---- | --------- | ------------ |
| 50257      | False           | True    | BF16  | 32          | 1024 | 352.8        | 92892  |
| 50257      | True            | True    | BF16  | 32          | 1024 | 280.4        | 116856.3  |

## Vocab Size
| Vocab Size | Flash Attention | Compile | dType | Batch Size  | T    | DT (msec) | Tokens / Sec |
| ---------- | --------------- | ------- | ----- | ----------  | ---- | --------- | ------------ |
| 50257      | False           | True    | BF16  | 32          | 1024 | 352.8        | 92892  |
| 50304      | True            | True    | BF16  | 32          | 1024 | 214.4        |  152857.6 |