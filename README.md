# Verilog-CNN
## RUN
```
vcs -full64 -cpp g++-4.8 -cc gcc-4.8 -LDFLAGS -Wl,-no-as-needed  -sverilog -debug_access+all ./CNN_tb.v ./CNN.v ./conv.v  ./conv_buffer.v ./conv_cal.v  ./maxpool_relu.v ./maxpool.v ./fc.v   +incdir+./ -R 
```
