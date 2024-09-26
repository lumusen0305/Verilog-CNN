module conv #(parameter WIDTH = 28, HEIGHT = 28,FILTER_SIZE = 5, DATA_BITS = 8,CHANNEL_LEN = 3)(
    input clk,
    input in_val,
    input rst_n,
    input [DATA_BITS - 1:0] data_in,
    input  [FILTER_SIZE *FILTER_SIZE *DATA_BITS*CHANNEL_LEN - 1:0] l1_weight,    
    input  [CHANNEL_LEN *DATA_BITS - 1:0] l1_bias,
    output [CHANNEL_LEN*(DATA_BITS)-1:0] data_out,
    output out_val
);
wire  [(FILTER_SIZE*FILTER_SIZE)*DATA_BITS - 1:0] data_out_b1;
wire val_b1;

conv_buffer #(.WIDTH(WIDTH), .HEIGHT(HEIGHT), .DATA_BITS(DATA_BITS), .FILTER_SIZE(FILTER_SIZE))
conv_buffer_l1(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(in_val),
    .data_in(data_in),
    .data_out(data_out_b1),
    .valid(val_b1)
);
conv_calc #(.FILTER_SIZE(FILTER_SIZE), .DATA_BITS(DATA_BITS), .CHANNEL_LEN(CHANNEL_LEN))
conv_calc_l1(
    .clk(clk),
    .in_val(val_b1),
    .rst_n(rst_n),
    .data_in(data_out_b1),
    .weight(l1_weight),
    .bias(l1_bias),
    .data_out(data_out),
    .valid(out_val)
);

endmodule