These two examples illustrate simple simulation of a digital BPSK
modulated communication system where only one sample per symbol is used,
and signal is affected only by AWGN noise.

In the first example, we cycle through different signal to noise values,
and the signal length is a function of theoretical probability of error.
As a rule of thumb, we want to count about 100 errors for each SNR
value, which determines the length of the signal (and noise) vector(s).



.. code-block:: python

    #!/usr/bin/python
    # BPSK digital modulation example
    # by Ivo Maljevic
    
    from numpy import *
    from scipy.special import erfc
    import matplotlib.pyplot as plt
    
    SNR_MIN     = 0
    SNR_MAX     = 9
    Eb_No_dB    = arange(SNR_MIN,SNR_MAX+1)
    SNR         = 10**(Eb_No_dB/10.0)  # linear SNR
    
    Pe          = empty(shape(SNR))
    BER         = empty(shape(SNR))
    
    loop = 0
    for snr in SNR:      # SNR loop
     Pe[loop] = 0.5*erfc(sqrt(snr))
     VEC_SIZE = ceil(100/Pe[loop])  # vector length is a function of Pe
    
     # signal vector, new vector for each SNR value
     s = 2*random.randint(0,high=2,size=VEC_SIZE)-1
    
     # linear power of the noise; average signal power = 1
     No = 1.0/snr
    
     # noise
     n = sqrt(No/2)*random.randn(VEC_SIZE)
    
     # signal + noise
     x = s + n
    
     # decode received signal + noise
     y = sign(x)
    
     # find erroneous symbols
     err = where(y != s)
     error_sum = float(len(err[0]))
     BER[loop] = error_sum/VEC_SIZE
     print 'Eb_No_dB=%4.2f, BER=%10.4e, Pe=%10.4e' % \
            (Eb_No_dB[loop], BER[loop], Pe[loop])
     loop += 1
    
    #plt.semilogy(Eb_No_dB, Pe,'r',Eb_No_dB, BER,'s')
    plt.semilogy(Eb_No_dB, Pe,'r',linewidth=2)
    plt.semilogy(Eb_No_dB, BER,'-s')
    plt.grid(True)
    plt.legend(('analytical','simulation'))
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('BER')
    plt.show()
    



In the second, slightly modified example, the problem of signal length
growth is solved by braking a signal into frames.Namely, the number of
samples for a given SNR grows quickly, so that the simulation above is
not practical for Eb/No values greater than 9 or 10 dB.



.. code-block:: python

    
    #!/usr/bin/python
    # BPSK digital modulation: modified example
    # by Ivo Maljevic
    
    from scipy import *
    from math import sqrt, ceil  # scalar calls are faster
    from scipy.special import erfc
    import matplotlib.pyplot as plt
    
    rand   = random.rand
    normal = random.normal
    
    SNR_MIN   = 0
    SNR_MAX   = 10
    FrameSize = 10000
    Eb_No_dB  = arange(SNR_MIN,SNR_MAX+1)
    Eb_No_lin = 10**(Eb_No_dB/10.0)  # linear SNR
    
    # Allocate memory
    Pe        = empty(shape(Eb_No_lin))
    BER       = empty(shape(Eb_No_lin))
    
    # signal vector (for faster exec we can repeat the same frame)
    s = 2*random.randint(0,high=2,size=FrameSize)-1
    
    loop = 0
    for snr in Eb_No_lin:
     No        = 1.0/snr
     Pe[loop]  = 0.5*erfc(sqrt(snr))
     nFrames   = ceil(100.0/FrameSize/Pe[loop])
     error_sum = 0
     scale = sqrt(No/2)
    
     for frame in arange(nFrames):
       # noise
       n = normal(scale=scale, size=FrameSize)
    
       # received signal + noise
       x = s + n
    
       # detection (information is encoded in signal phase)
       y = sign(x)
    
       # error counting
       err = where (y != s)
       error_sum += len(err[0])
    
       # end of frame loop
       ##################################################
    
     BER[loop] = error_sum/(FrameSize*nFrames)  # SNR loop level
     print 'Eb_No_dB=%2d, BER=%10.4e, Pe[loop]=%10.4e' % \
            (Eb_No_dB[loop], BER[loop], Pe[loop])
     loop += 1
    
    plt.semilogy(Eb_No_dB, Pe,'r',linewidth=2)
    plt.semilogy(Eb_No_dB, BER,'-s')
    plt.grid(True)
    plt.legend(('analytical','simulation'))
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('BER')
    plt.show()
    



.. image:: CommTheory_attachments/BPSK_BER.PNG

--------------

``. CategoryCookbook``

