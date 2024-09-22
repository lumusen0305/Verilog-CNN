module maxpool_relu #(parameter WIDTH = 28, HEIGHT = 28,FILTER_SIZE = 5, DATA_BITS = 8)(
    input clk,
    input in_val,
    input rst_n,
    input [DATA_BITS - 1:0] data_in,
    output signed [DATA_BITS - 1:0] data_out,
    output valid
);
wire  [(FILTER_SIZE*FILTER_SIZE)*DATA_BITS - 1:0] data_out_b1;
conv_buffer_2w #(.WIDTH(WIDTH), .HEIGHT(HEIGHT), .DATA_BITS(DATA_BITS), .FILTER_SIZE(FILTER_SIZE))
conv_buffer_l1(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(in_val),
    .data_in(data_in),
    .data_out(data_out_b1),
    .valid(valid)
);
maxpool #(.FILTER_SIZE(FILTER_SIZE),.DATA_BITS(DATA_BITS))
maxpool_u1 (
    .data_in(data_out_b1), 
    .data_out(data_out)
);
endmodule