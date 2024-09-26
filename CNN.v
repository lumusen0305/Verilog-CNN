module CNN(
    input clk,
    input in_val,
    input rst_n,
    input  [32 - 1:0] data_in,
    input  [5 *5 *32*3 - 1:0] l1_weight,
    input  [5 *5 *32*3 - 1:0] l2_c1_weight,    
    input  [5 *5 *32*3 - 1:0] l2_c2_weight,    
    input  [5 *5 *32*3 - 1:0] l2_c3_weight,    
    input  [10 *48*32 - 1:0] fc_weight,    
    input  [3 *32 - 1:0] l1_bias,
    input  [3 *32 - 1:0] l2_bias,
    input  [10 *32 - 1:0] fc_bias,
    
    output [3:0] decision,
    output out_val
);
parameter FILTER_SIZE = 5, DATA_BITS = 32,CHANNEL_LEN = 3;

wire  [CHANNEL_LEN*(DATA_BITS) - 1:0] data_out_c1;
wire  [CHANNEL_LEN*(DATA_BITS) - 1:0] data_out_l2_c1;
wire  [CHANNEL_LEN*(DATA_BITS) - 1:0] data_out_l2_c2;
wire  [CHANNEL_LEN*(DATA_BITS) - 1:0] data_out_l2_c3;

wire  [(DATA_BITS) - 1:0] data_out_l1_c1,data_out_l1_c2,data_out_l1_c3;
wire val_c1;
wire val_l1_m1,val_l1_m2,val_l1_m3;
wire val_l2_m1,val_l2_m2,val_l2_m3;
wire val_l2_c1,val_l2_c2,val_l2_c3;

        
conv  #(.WIDTH(28), .HEIGHT(28), .DATA_BITS(DATA_BITS), .FILTER_SIZE(5))
conv1_bias( 
    .clk(clk),
    .rst_n(rst_n),
    .in_val(in_val),
    .data_in(data_in),
    .l1_weight(l1_weight),
    .l1_bias(l1_bias),
    .data_out(data_out_c1),
    .out_val(val_c1)
);  
// ====================L1 MAX POOL================== 
maxpool_relu #(.WIDTH(24), .HEIGHT(24), .DATA_BITS((DATA_BITS)), .FILTER_SIZE(2))
maxpool_relu_l1_c1(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(val_c1),
    .data_in(data_out_c1[0 +:(DATA_BITS)]),
    .data_out(data_out_l1_c1),
    .valid(val_l1_m1)
);                      
maxpool_relu #(.WIDTH(24), .HEIGHT(24), .DATA_BITS((DATA_BITS)), .FILTER_SIZE(2))
maxpool_relu_l1_c2(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(val_c1),
    .data_in(data_out_c1[1*(DATA_BITS) +:(DATA_BITS)]),
    .data_out(data_out_l1_c2),
    .valid(val_l1_m2)
);
maxpool_relu #(.WIDTH(24), .HEIGHT(24), .DATA_BITS((DATA_BITS)), .FILTER_SIZE(2))
maxpool_relu_l1_c3(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(val_c1),
    .data_in(data_out_c1[2*(DATA_BITS)  +:(DATA_BITS)]),
    .data_out(data_out_l1_c3),
    .valid(val_l1_m3)
);

// ====================L1 MAX POOL================== 
// ====================  L2 CONV  ================== 

conv  #(.WIDTH(12), .HEIGHT(12), .DATA_BITS(DATA_BITS), .FILTER_SIZE(5))
conv2_bias_c1( 
    .clk(clk),
    .rst_n(rst_n),
    .in_val(val_l1_m1),
    .data_in(data_out_l1_c1),
    .l1_weight(l2_c1_weight),
    .l1_bias(0),
    .data_out(data_out_l2_c1),
    .out_val(val_l2_c1)
);


conv  #(.WIDTH(12), .HEIGHT(12), .DATA_BITS(DATA_BITS), .FILTER_SIZE(5))
conv2_bias_c2( 
    .clk(clk),
    .rst_n(rst_n),
    .in_val(val_l1_m2),
    .data_in(data_out_l1_c2),
    .l1_weight(l2_c2_weight),
    .l1_bias(0),
    .data_out(data_out_l2_c2),
    .out_val(val_l2_c2)
);


conv  #(.WIDTH(12), .HEIGHT(12), .DATA_BITS(DATA_BITS), .FILTER_SIZE(5))
conv2_bias_c3( 
    .clk(clk),
    .rst_n(rst_n),
    .in_val(val_l1_m3),
    .data_in(data_out_l1_c3),
    .l1_weight(l2_c3_weight),
    .l1_bias(0),
    .data_out(data_out_l2_c3),
    .out_val(val_l2_c3)
);

// ====================  L2 CONV  ================== 

// ====================L2 DATA PE ================== 
wire signed [DATA_BITS-1:0] l2_conv1_out= $signed(data_out_l2_c1[0 +: DATA_BITS])+$signed(data_out_l2_c1[1*DATA_BITS +: DATA_BITS])+$signed(data_out_l2_c1[2*DATA_BITS +: DATA_BITS])+$signed(l2_bias[0+:DATA_BITS]          );
wire signed [DATA_BITS-1:0] l2_conv2_out= $signed(data_out_l2_c2[0 +: DATA_BITS])+$signed(data_out_l2_c2[1*DATA_BITS +: DATA_BITS])+$signed(data_out_l2_c2[2*DATA_BITS +: DATA_BITS])+$signed(l2_bias[DATA_BITS+:DATA_BITS]  );
wire signed [DATA_BITS-1:0] l2_conv3_out= $signed(data_out_l2_c3[0 +: DATA_BITS])+$signed(data_out_l2_c3[1*DATA_BITS +: DATA_BITS])+$signed(data_out_l2_c3[2*DATA_BITS +: DATA_BITS])+$signed(l2_bias[2*DATA_BITS+:DATA_BITS]);
wire signed [DATA_BITS-1:0] l2_maxpool1_out;
wire signed [DATA_BITS-1:0] l2_maxpool2_out;
wire signed [DATA_BITS-1:0] l2_maxpool3_out;
// ====================L2 DATA PE ================== 

// ====================L2 MAX POOL================== 
maxpool_relu #(.WIDTH(8), .HEIGHT(8), .DATA_BITS(DATA_BITS), .FILTER_SIZE(2))
maxpool_relu_l2_m1(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(val_l2_c1),
    .data_in(l2_conv1_out),
    .data_out(l2_maxpool1_out),
    .valid(val_l2_m1)
);      
maxpool_relu #(.WIDTH(8), .HEIGHT(8), .DATA_BITS(DATA_BITS), .FILTER_SIZE(2))
maxpool_relu_l2_m2(
    .clk(clk),
    .rst_n(rst_n),
    .in_val(val_l2_c2),
    .data_in(l2_conv2_out),
    .data_out(l2_maxpool2_out),
    .valid(val_l2_m2)
);
maxpool_relu #(.WIDTH(8), .HEIGHT(8), .DATA_BITS(DATA_BITS), .FILTER_SIZE(2))
maxpool_relu_l2_m3(
    .clk(clk),      
    .rst_n(rst_n),
    .in_val(val_l2_c3),
    .data_in(l2_conv3_out),
    .data_out(l2_maxpool3_out),
    .valid(val_l2_m3)
);
// ====================L2 MAX POOL================== 
wire [3*(DATA_BITS)-1:0] l2_maxpool_out={l2_maxpool3_out,l2_maxpool2_out,l2_maxpool1_out};
// ==================Fully Connected================ 
fc #(.DATA_NUM(10), .DATA_BITS(DATA_BITS),.PARAMETER_DATA_BITS((DATA_BITS)), .CHANNEL_LEN(3))
fc_final(
    .clk(clk),
    .in_val(val_l2_m3),
    .rst_n(rst_n),
    .data_in(l2_maxpool_out),
    .bias(fc_bias),
    .weight(fc_weight),
    .decision(decision),
    .valid(out_val)
);

// ==================Fully Connected================ 


endmodule