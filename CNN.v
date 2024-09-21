module CNN(
    input clk,
    input in_val,
    input rst_n,
    input [8 - 1:0] data_in,
    input  [5 *5 *8*3 - 1:0] l1_weight,    
    input  [3 *8 - 1:0] l1_bias,
    output [3:0] decision,
    output out_val
);
parameter FILTER_SIZE = 5, DATA_BITS = 8,CHANNEL_LEN = 3;

wire  [(FILTER_SIZE*FILTER_SIZE)*DATA_BITS - 1:0] data_out_b1,data_out_c1;
wire val_b1,val_c1;



conv_buffer #(.WIDTH(28), .HEIGHT(28), .DATA_BITS(8), .FILTER_SIZE(5))
conv_buffer_l1(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(in_val),
    .data_in(data_in),
    .data_out(data_out_b1),
    .valid(val_b1)
);
conv_calc #(.FILTER_SIZE(5), .DATA_BITS(8), .CHANNEL_LEN(3))
conv_calc_l1(
    .clk(clk),
    .in_val(val_b1),
    .rst_n(rst_n),
    .data_in(data_out_b1),
    .weight(l1_weight),
    .bias(l1_bias),
    .data_out(data_out_c1),
    .valid(val_c1)
);

endmodule