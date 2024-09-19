module CNN(
    input clk,
    input in_val,
    input rst_n,
    input [8 - 1:0] data_in,
    output [3:0] decision,
    output out_val
);
wire  [(5*5-1)*8 - 1:0] data_out;

conv_buffer #(.WIDTH(28), .HEIGHT(28), .DATA_BITS(8), .FILTER_SIZE(5))
conv_buffer_l1(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(in_val),
    .data_in(data_in),
    .data_out(data_out),
    .valid(out_val)
);
endmodule