import numpy as np

def normalize(X, Y):
    Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
    Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)
    
    Xstd[0:24] = Xstd[0:24].mean()
    Xstd[24:108] = Xstd[24:108].mean()
    Xstd[108:192] = Xstd[108:192].mean()
    Xstd[192:304] = Xstd[192:304].mean()

    Ystd[0:14] = Ystd[0:14].mean()
    Ystd[14:98] = Ystd[14:98].mean()
    Ystd[98:182] = Ystd[98:182].mean()
    Ystd[182:294] = Ystd[182:294].mean()

    Xstd=np.where(Xstd!=0,Xstd,1)
    Ystd=np.where(Ystd!=0,Ystd,1)

    """ Save Mean / Std / Min / Max """
    Xmean.astype(np.float32).tofile('training/Xmean.bin')
    Ymean.astype(np.float32).tofile('training/Ymean.bin')
    Xstd.astype(np.float32).tofile('training/Xstd.bin')
    Ystd.astype(np.float32).tofile('training/Ystd.bin')

    """ Normalize Data """
    X = (X - Xmean) / Xstd
    Y = (Y - Ymean) / Ystd
    return X, Y

def Run_denormalize_Y(Y):
    Ymean = np.fromfile('training/Ymean.bin',dtype=np.float32)
    Ystd = np.fromfile('training/Ystd.bin',dtype=np.float32)
    # print(Ystd.shape)
    return (Y*Ystd)+Ymean

def Run_denormalize_X(X):
    Xmean = np.fromfile('training/Xmean.bin',dtype=np.float32)
    Xstd = np.fromfile('training/Xstd.bin',dtype=np.float32)

    return (X*Xstd)+Xmean

def Run_normalize_X(X):
    Xmean = np.fromfile('training/Xmean.bin',dtype=np.float32)
    Xstd = np.fromfile('training/Xstd.bin',dtype=np.float32)

    return (X - Xmean) / Xstd
