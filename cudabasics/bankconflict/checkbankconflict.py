import os
import sys


print(__name__)
if __name__ == "__main__":
    print("Checking bank conflict")
    BN = 128
    BM = 128
    BK = 16
    NUMTHREADS = 128
    ROWSTRIDE = NUMTHREADS // (BK // 4)
    PAD = 1
    i = 0
    while i < BM:
        for threadid in range(128):
            colA = threadid % (BK // 4) 
            rowA = threadid // (BK // 4)
            # lane = threadid % 32
            # lane = colA
            # rowA = (lane % 32) ^ (lane // 32)
            # As[(innerColA * 4 + 0) * BM + innerRowA + i]
            # address = ((colA * 4 + 0) * BM + rowA + i)
            # address = (rowA + i) * BK + colA
            address = ((colA * 4 + 0) * (BM + PAD) + rowA + i)

            # address = (address % 32) ^ (address // 32)

            print(f"thread {threadid}: col={colA}, row={rowA}, adddress={address}, bank={address % 32}")

        # for threadid in range(128):
        #     colB = threadid % (BN // 4) 
        #     rowB = threadid // (BN // 4)
        #     # Bs + (innerRowB + i) * BN + innerColB * 4
        #     address = ((rowB + i) * BN + colB)
        #     bank = address  % 32
        #     print(f"thread {threadid}: col={colB}, row={rowB}, adddress={address}, bank={bank}")
        i += ROWSTRIDE 