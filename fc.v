module fc #(parameter DATA_NUM = 10, DATA_BITS = 8,PARAMETER_DATA_BITS = 8, CHANNEL_LEN = 3)(
    input clk,
    input in_val,
    input rst_n,
    input signed [CHANNEL_LEN*DATA_BITS - 1:0] data_in,
    input signed [DATA_NUM*PARAMETER_DATA_BITS - 1:0] bias,
    input signed [48*DATA_NUM*PARAMETER_DATA_BITS - 1:0] weight ,
    output reg [3:0] decision,
    output reg valid
);
    parameter DATA_MEM=DATA_BITS;
    reg [(DATA_NUM)*DATA_MEM-1:0] fc_mem;
    reg [5:0] index;

    reg signed [DATA_BITS-1:0] max_value;  // 存储最大值
    reg [3:0] j;  // 循环变量，用于遍历 fc_mem
    
    always @(*) begin
        max_value = fc_mem[DATA_MEM-1:0];  // 初始化为第一个数据块的值
        decision = 0;  // 初始化最大值索引为 0
        for (j = 1; j < DATA_NUM; j = j + 1) begin
            if ($signed(fc_mem[j * DATA_MEM +: DATA_MEM]) > max_value) begin
                max_value = fc_mem[j * DATA_MEM +: DATA_MEM];  // 更新最大值
                decision = j;  // 更新最大值对应的索引
            end
        end
    end

    genvar i;
    generate
        for (i = 0; i < DATA_NUM; i = i + 1) begin : gen_reset
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    fc_mem[i*DATA_MEM +: DATA_MEM] <= $signed(bias[i*DATA_BITS +: DATA_BITS]);
                end
                else begin
                    if (in_val) begin
                    fc_mem[i*DATA_MEM +: DATA_MEM] <= $signed(fc_mem[i*DATA_MEM +: DATA_MEM])
                        + $signed(data_in[0+:DATA_BITS]) * $signed(weight[i*index*PARAMETER_DATA_BITS+:PARAMETER_DATA_BITS])
                        + $signed(data_in[DATA_BITS+:DATA_BITS]) * $signed(weight[i*index*PARAMETER_DATA_BITS+:PARAMETER_DATA_BITS])
                        + $signed(data_in[DATA_BITS*2+:DATA_BITS]) * $signed(weight[i*2*index*PARAMETER_DATA_BITS+:PARAMETER_DATA_BITS]);
                        end
                end
            end
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            index<=0;
            valid<=0;
        end
        else begin
            if (index<16) begin                         
                valid<=0;
                if(in_val)index<=index+1;
            end
            else begin
                valid<=1;
            end
        end
    end
endmodule