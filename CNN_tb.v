module CNN_tb();
reg clk, rst_n;
reg [7:0] pixels [0:783];
reg [9:0] img_idx;
reg [7:0] data_in;
wire [3:0] decision;
reg valid,valid_o;

  parameter FILTER_SIZE = 5, DATA_BITS = 8,CHANNEL_LEN = 3;

  // Define weight and bias arrays
  reg signed [DATA_BITS - 1:0] l1_weight_1 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l1_weight_2 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l1_weight_3 [0:FILTER_SIZE * FILTER_SIZE - 1];
  reg signed [DATA_BITS - 1:0] l1_bias [0:CHANNEL_LEN - 1];
  
 initial begin
   $readmemh("./parameter/conv1_weight_1.txt", l1_weight_1);
   $readmemh("./parameter/conv1_weight_2.txt", l1_weight_2);
   $readmemh("./parameter/conv1_weight_3.txt", l1_weight_3);
   $readmemh("./parameter/conv1_bias.txt", l1_bias);
 end

  // Concatenate all weights into a single wire
  wire [FILTER_SIZE * FILTER_SIZE * DATA_BITS * CHANNEL_LEN - 1:0] l1_weight;
  wire [CHANNEL_LEN * DATA_BITS - 1:0] l1_bias_concat;
  assign l1_bias_concat = {l1_bias[2], l1_bias[1], l1_bias[0]}; // 拼接成位串

  genvar i;
  generate
    for (i = 0; i < FILTER_SIZE * FILTER_SIZE; i = i + 1) begin : gen_weights
      assign l1_weight[ i * DATA_BITS +: DATA_BITS] = l1_weight_1[i];
      assign l1_weight[(i * DATA_BITS + DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] = l1_weight_2[i];
      assign l1_weight[(i * DATA_BITS + 2 * DATA_BITS * FILTER_SIZE * FILTER_SIZE) +: DATA_BITS] = l1_weight_3[i];
    end
  endgenerate
CNN dut(
    .clk(clk),
    .rst_n(rst_n),
    .l1_weight(l1_weight),
    .l1_bias(l1_bias_concat),
    .in_val(valid),
    .data_in(data_in),
    .decision(),
    .out_val(valid_o)
  );
// Clock generation
always #5 clk = ~clk;
// Read image text file
initial begin
  $readmemh("3_0.txt", pixels);
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
    end else #10 $finish; 
    data_in <= pixels[img_idx];
    valid<=1;
  end
end
endmodule