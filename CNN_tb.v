module CNN_tb();
reg clk, rst_n;
reg [7:0] pixels [0:783];
reg [9:0] img_idx;
reg [7:0] data_in;
wire [3:0] decision;
reg valid,valid_o;
CNN dut(
    .clk(clk),
    .rst_n(rst_n),
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