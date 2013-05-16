These are quick examples of using
`ctypes <http://docs.python.org/lib/module-ctypes.html>`__ and numpy to
do data acquisition and playback using `National
Instrument's <http://ni.com>`__
`NI-DAQmx <http://www.ni.com/dataacquisition/nidaqmx.htm>`__ library.
This library allows access to their wide range of data acquisition
devices. By using ctypes, we bypass the need for a C compiler. The code
below assumes a Windows platform. NI-DAQmx is also available for Linux,
but the code below would require a few minor changes, namely loading the
shared library and setting the function signatures.

See also [:Cookbook/Data Acquisition with PyUL:Data acquisition with
PyUniversalLibrary].

See also projects that wrap NI-DAQmx library with Linux support:
`pylibnidaqmx <http://code.google.com/p/pylibnidaqmx/>`__,
`pydaqmx <http://code.google.com/p/pydaqmx/>`__,
`daqmxbase-swig <http://code.google.com/p/daqmxbase-swig/>`__.

OK, enough talk, let's see the code!

Analog Acquisition
==================




.. code-block:: python

    #!python numbers=disable
    #Acq_IncClk.py
    # This is a near-verbatim translation of the example program
    # C:\Program Files\National Instruments\NI-DAQ\Examples\DAQmx ANSI C\Analog In\M
    #easure Voltage\Acq-Int Clk\Acq-IntClk.c
    import ctypes
    import numpy
    nidaq = ctypes.windll.nicaiu # load the DLL
    ##############################
    # Setup some typedefs and constants
    # to correspond with values in
    # C:\Program Files\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include\NIDAQmx.
    #h
    # the typedefs
    int32 = ctypes.c_long
    uInt32 = ctypes.c_ulong
    uInt64 = ctypes.c_ulonglong
    float64 = ctypes.c_double
    TaskHandle = uInt32
    # the constants
    DAQmx_Val_Cfg_Default = int32(-1)
    DAQmx_Val_Volts = 10348
    DAQmx_Val_Rising = 10280
    DAQmx_Val_FiniteSamps = 10178
    DAQmx_Val_GroupByChannel = 0
    ##############################
    def CHK(err):
        """a simple error checking routine"""
        if err < 0:
            buf_size = 100
            buf = ctypes.create_string_buffer('\000' * buf_size)
            nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
            raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(buf.v
    alue)))
    # initialize variables
    taskHandle = TaskHandle(0)
    max_num_samples = 1000
    data = numpy.zeros((max_num_samples,),dtype=numpy.float64)
    # now, on with the program
    CHK(nidaq.DAQmxCreateTask("",ctypes.byref(taskHandle)))
    CHK(nidaq.DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai0","",
                                       DAQmx_Val_Cfg_Default,
                                       float64(-10.0),float64(10.0),
                                       DAQmx_Val_Volts,None))
    CHK(nidaq.DAQmxCfgSampClkTiming(taskHandle,"",float64(10000.0),
                                    DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,
                                    uInt64(max_num_samples)));
    CHK(nidaq.DAQmxStartTask(taskHandle))
    read = int32()
    CHK(nidaq.DAQmxReadAnalogF64(taskHandle,max_num_samples,float64(10.0),
                                 DAQmx_Val_GroupByChannel,data.ctypes.data,
                                 max_num_samples,ctypes.byref(read),None))
    print "Acquired %d points"%(read.value)
    if taskHandle.value != 0:
        nidaq.DAQmxStopTask(taskHandle)
        nidaq.DAQmxClearTask(taskHandle)
    print "End of program, press Enter key to quit"
    raw_input()
    



Analog Generation
=================




.. code-block:: python

    #!python numbers=disable
    """
    This is an interpretation of the example program
    C:\Program Files\National Instruments\NI-DAQ\Examples\DAQmx ANSI C\Analog Out\Ge
    nerate Voltage\Cont Gen Volt Wfm-Int Clk\ContGen-IntClk.c
    This routine will play an arbitrary-length waveform file.
    This module depends on:
    numpy
    Adapted by Martin Bures [ mbures { @ } zoll { . } com ]
    """
    # import system libraries
    import ctypes
    import numpy
    import threading
    # load any DLLs
    nidaq = ctypes.windll.nicaiu # load the DLL
    ##############################
    # Setup some typedefs and constants
    # to correspond with values in
    # C:\Program Files\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include\NIDAQmx.
    #h
    # the typedefs
    int32 = ctypes.c_long
    uInt32 = ctypes.c_ulong
    uInt64 = ctypes.c_ulonglong
    float64 = ctypes.c_double
    TaskHandle = uInt32
    # the constants
    DAQmx_Val_Cfg_Default = int32(-1)
    DAQmx_Val_Volts = 10348
    DAQmx_Val_Rising = 10280
    DAQmx_Val_FiniteSamps = 10178
    DAQmx_Val_ContSamps = 10123
    DAQmx_Val_GroupByChannel = 0
    ##############################
    class WaveformThread( threading.Thread ):
        """
        This class performs the necessary initialization of the DAQ hardware and
        spawns a thread to handle playback of the signal.
        It takes as input arguments the waveform to play and the sample rate at whic
    h
        to play it.
        This will play an arbitrary-length waveform file.
        """
        def __init__( self, waveform, sampleRate ):
            self.running = True
            self.sampleRate = sampleRate
            self.periodLength = len( waveform )
            self.taskHandle = TaskHandle( 0 )
            self.data = numpy.zeros( ( self.periodLength, ), dtype=numpy.float64 )
            # convert waveform to a numpy array
            for i in range( self.periodLength ):
                self.data[ i ] = waveform[ i ]
            # setup the DAQ hardware
            self.CHK(nidaq.DAQmxCreateTask("",
                              ctypes.byref( self.taskHandle )))
            self.CHK(nidaq.DAQmxCreateAOVoltageChan( self.taskHandle,
                                       "Dev1/ao0",
                                       "",
                                       float64(-10.0),
                                       float64(10.0),
                                       DAQmx_Val_Volts,
                                       None))
            self.CHK(nidaq.DAQmxCfgSampClkTiming( self.taskHandle,
                                    "",
                                    float64(self.sampleRate),
                                    DAQmx_Val_Rising,
                                    DAQmx_Val_FiniteSamps,
                                    uInt64(self.periodLength)));
            self.CHK(nidaq.DAQmxWriteAnalogF64( self.taskHandle,
                                  int32(self.periodLength),
                                  0,
                                  float64(-1),
                                  DAQmx_Val_GroupByChannel,
                                  self.data.ctypes.data,
                                  None,
                                  None))
            threading.Thread.__init__( self )
        def CHK( self, err ):
            """a simple error checking routine"""
            if err < 0:
                buf_size = 100
                buf = ctypes.create_string_buffer('\000' * buf_size)
                nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
                raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(b
    uf.value)))
            if err > 0:
                buf_size = 100
                buf = ctypes.create_string_buffer('\000' * buf_size)
                nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
                raise RuntimeError('nidaq generated warning %d: %s'%(err,repr(buf.va
    lue)))
        def run( self ):
            counter = 0
            self.CHK(nidaq.DAQmxStartTask( self.taskHandle ))
        def stop( self ):
            self.running = False
            nidaq.DAQmxStopTask( self.taskHandle )
            nidaq.DAQmxClearTask( self.taskHandle )
    if __name__ == '__main__':
        import time
        # generate a time signal 5 seconds long with 250Hz sample rate
        t = numpy.arange( 0, 5, 1.0/250.0 )
        # generate sine wave
        x = sin( t )
        mythread = WaveformThread( x, 250 )
        # start playing waveform
        mythread.start()
        # wait 5 seconds then stop
        time.sleep( 5 )
        mythread.stop()
    



CategoryCookbook

