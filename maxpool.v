module maxpool #(parameter FILTER_SIZE = 5, DATA_BITS = 8)(
    input [(FILTER_SIZE * FILTER_SIZE) * DATA_BITS - 1:0] data_in, // 輸入的數據
    output reg signed [DATA_BITS - 1:0] data_out  // 最大值輸出
);

    // 定義內部變量來保存展開的數據
    reg signed [DATA_BITS - 1:0] exp_data [0:FILTER_SIZE * FILTER_SIZE - 1];
    reg signed [DATA_BITS - 1:0] col_max [0:FILTER_SIZE - 1];  // 每列的最大值
    reg signed [DATA_BITS - 1:0] global_max;  // 全局最大值

    // 使用 generate 語句將 data_in 展開成多個數據
    genvar i;
    generate
        for (i = 0; i < FILTER_SIZE * FILTER_SIZE; i = i + 1) begin : unpack_data
            always @(*) begin
                exp_data[i] = data_in[i * DATA_BITS +: DATA_BITS];
            end
        end
    endgenerate

    // 第一步：按列比較
    integer j, k;
    always @(*) begin
        for (k = 0; k < FILTER_SIZE; k = k + 1) begin
            col_max[k] = exp_data[k];  // 初始化為該列的第一個元素
            for (j = 1; j < FILTER_SIZE; j = j + 1) begin
                if (exp_data[k + j * FILTER_SIZE] > col_max[k]) begin
                    col_max[k] = exp_data[k + j * FILTER_SIZE];  // 更新該列的最大值
                end
            end
        end
    end

    // 第二步：對所有列的最大值進行比較（使用 always @(*) ）
    integer l;
    always @(*) begin
        global_max = col_max[0];  // 初始化為第一列的最大值
        for (l = 1; l < FILTER_SIZE; l = l + 1) begin
            if (col_max[l] > global_max) begin
                global_max = col_max[l];  // 更新全局最大值
            end
        end
        // relu 操作，如果最大值小於 0，輸出 0
        data_out = (global_max > 0) ? global_max : 0;  
    end

endmodule