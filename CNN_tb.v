module CNN_tb();
reg clk, rst_n;
reg [31:0] pixels [0:783];
reg [9:0] img_idx;
reg [31:0] data_in;
wire [3:0] decision;
reg valid,valid_o;

  parameter FILTER_SIZE = 5, DATA_BITS = 32,CHANNEL_LEN = 3,INPUT_NUM = 48, OUTPUT_NUM = 10;

  // Define weight and bias arrays
  reg signed [DATA_BITS - 1:0] l1_weight_1 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l1_weight_2 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l1_weight_3 [0:FILTER_SIZE * FILTER_SIZE - 1];

  reg signed [DATA_BITS - 1:0] l2_weight_c1_1 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l2_weight_c1_2 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l2_weight_c1_3 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l2_weight_c2_1 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l2_weight_c2_2 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l2_weight_c2_3 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l2_weight_c3_1 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l2_weight_c3_2 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l2_weight_c3_3 [0:FILTER_SIZE * FILTER_SIZE - 1];

  reg signed [DATA_BITS - 1:0] l1_bias [0:CHANNEL_LEN - 1];
  reg signed [DATA_BITS - 1:0] l2_bias [0:CHANNEL_LEN - 1];
  reg signed [DATA_BITS - 1:0] fc_weight_temp [0:INPUT_NUM * OUTPUT_NUM - 1];
  reg signed [DATA_BITS - 1:0] fc_bias_temp [0:OUTPUT_NUM - 1];

 initial begin
   $readmemh("./parameter/conv1_weight_1.txt", l1_weight_1);
   $readmemh("./parameter/conv1_weight_2.txt", l1_weight_2);
   $readmemh("./parameter/conv1_weight_3.txt", l1_weight_3);

   $readmemh("./parameter/conv2_weight_11.txt", l2_weight_c1_1);
   $readmemh("./parameter/conv2_weight_12.txt", l2_weight_c1_2);
   $readmemh("./parameter/conv2_weight_13.txt", l2_weight_c1_3);
   $readmemh("./parameter/conv2_weight_21.txt", l2_weight_c2_1);
   $readmemh("./parameter/conv2_weight_22.txt", l2_weight_c2_2);
   $readmemh("./parameter/conv2_weight_23.txt", l2_weight_c2_3);
   $readmemh("./parameter/conv2_weight_31.txt", l2_weight_c3_1);
   $readmemh("./parameter/conv2_weight_32.txt", l2_weight_c3_2);
   $readmemh("./parameter/conv2_weight_33.txt", l2_weight_c3_3);
   $readmemh("./parameter/fc_weight.txt", fc_weight_temp);
   $readmemh("./parameter/fc_bias.txt", fc_bias_temp);

   $readmemh("./parameter/conv1_bias.txt", l1_bias);
   $readmemh("./parameter/conv2_bias.txt", l2_bias);
 end

  // Concatenate all weights into a single wire
  wire [FILTER_SIZE * FILTER_SIZE * DATA_BITS * CHANNEL_LEN - 1:0] l1_weight,l2_c1_weight,l2_c2_weight,l2_c3_weight;
  
  wire  [10 *8 - 1:0] fc_bias;
  wire  [10 *48*8 - 1:0] fc_weight;

  wire [CHANNEL_LEN * DATA_BITS - 1:0] l1_bias_concat,l2_bias_concat;
  assign l1_bias_concat = {l1_bias[2], l1_bias[1], l1_bias[0]}; // 拼接成位串
  assign l2_bias_concat = {l2_bias[2], l2_bias[1], l2_bias[0]}; // 拼接成位串

  generate
      genvar j;
      for (j = 0; j < INPUT_NUM * OUTPUT_NUM; j = j + 1) begin : gen_fc_weights
          assign fc_weight[j * DATA_BITS +: DATA_BITS] = fc_weight_temp[j];
      end
  endgenerate

  assign fc_bias={
    fc_bias_temp[9],
    fc_bias_temp[8],
    fc_bias_temp[7],
    fc_bias_temp[6],
    fc_bias_temp[5],
    fc_bias_temp[4],
    fc_bias_temp[3],
    fc_bias_temp[2],
    fc_bias_temp[1],
    fc_bias_temp[0]}; 
  genvar i;
  generate
    for (i = 0; i < FILTER_SIZE * FILTER_SIZE; i = i + 1) begin : gen_weights
      assign l1_weight[ i * DATA_BITS +: DATA_BITS] = l1_weight_1[i];
      assign l1_weight[(i * DATA_BITS + DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] = l1_weight_2[i];
      assign l1_weight[(i * DATA_BITS + 2 * DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] = l1_weight_3[i];

      assign l2_c1_weight[ i * DATA_BITS +: DATA_BITS] =                                              l2_weight_c1_1[i];
      assign l2_c1_weight[(i * DATA_BITS + DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] =     l2_weight_c1_2[i];
      assign l2_c1_weight[(i * DATA_BITS + 2 * DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] = l2_weight_c1_3[i];
      assign l2_c2_weight[ i * DATA_BITS +: DATA_BITS] =                                              l2_weight_c2_1[i];
      assign l2_c2_weight[(i * DATA_BITS + DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] =     l2_weight_c2_2[i];
      assign l2_c2_weight[(i * DATA_BITS + 2 * DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] = l2_weight_c2_3[i];
      assign l2_c3_weight[ i * DATA_BITS +: DATA_BITS] =                                              l2_weight_c3_1[i];
      assign l2_c3_weight[(i * DATA_BITS + DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] =     l2_weight_c3_2[i];
      assign l2_c3_weight[(i * DATA_BITS + 2 * DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] = l2_weight_c3_3[i];
    end
  endgenerate
wire [3:0]decision;
CNN dut(
    .clk(clk),
    .rst_n(rst_n),
    .l1_weight(l1_weight),
    .l2_c1_weight(l2_c1_weight),
    .l2_c2_weight(l2_c2_weight),
    .l2_c3_weight(l2_c3_weight),
    .fc_weight(fc_weight),
    .l1_bias(l1_bias_concat),
    .l2_bias(l2_bias_concat),
    .fc_bias(fc_bias),
    .in_val(valid),
    .data_in(data_in),
    .decision(decision),
    .out_val(valid_o)
  );
// Clock generation
always #5 clk = ~clk;
// Read image text file
initial begin
  $readmemh("./test/3_0.txt", pixels);
  clk <= 1'b0;
  rst_n <= 1'b1;
  #3
  rst_n <= 1'b0;
  #3
  rst_n <= 1'b1;
end         
initial begin
  $fsdbDumpfile("wave.fsdb");
  $fsdbDumpvars(0, CNN_tb);
end   
always @(posedge clk) begin
  if(~rst_n) begin
    img_idx <= 0;
    valid<=0;
  end else begin
    if(img_idx < 10'd784) begin
      img_idx <= img_idx + 1'b1;
    end
    if (valid_o) begin
      $display("Decision:%d",decision);
      #10 $finish;
    end 
    data_in <= pixels[img_idx];
    valid<=1;
  end
end
endmodule