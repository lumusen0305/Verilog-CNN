    module conv_calc #(parameter FILTER_SIZE = 5, DATA_BITS = 8,CHANNEL_LEN = 3)(
        input clk,
        input in_val,
        input rst_n,
        input  signed [(FILTER_SIZE*FILTER_SIZE)*DATA_BITS - 1:0] data_in,
        input  signed [FILTER_SIZE * FILTER_SIZE * DATA_BITS*CHANNEL_LEN - 1:0] weight,    
        input  signed [CHANNEL_LEN * DATA_BITS - 1:0]bias,
        output signed [CHANNEL_LEN*(DATA_BITS)-1:0] data_out,
        output valid
    );

    parameter N = FILTER_SIZE * FILTER_SIZE; // 25
    parameter SHIFT_NUM=16;
    wire signed [CHANNEL_LEN * DATA_BITS - 1 : 0] calc_out;
    wire signed [DATA_BITS-1:0] exp_bias [0:CHANNEL_LEN - 1];
    wire signed [DATA_BITS-1:0] exp_data [0:FILTER_SIZE * FILTER_SIZE - 1];

    // Unsigned -> Signed
    genvar i;
    generate
        for (i = 0; i < FILTER_SIZE * FILTER_SIZE; i = i + 1) begin : gen_exp_data
            assign exp_data[i] = data_in[i * DATA_BITS +: DATA_BITS];
        end
    endgenerate

    genvar c;
    generate    
        for (c = 0; c < CHANNEL_LEN; c = c + 1) begin : gen_exp_bias
            assign exp_bias[c] =bias[c* DATA_BITS+:DATA_BITS];
        end
    endgenerate

    reg signed [DATA_BITS - 1:0] weight_array [0:CHANNEL_LEN - 1][0:FILTER_SIZE * FILTER_SIZE - 1];
    genvar d, w;
    generate
        for (d = 0; d < CHANNEL_LEN; d = d + 1) begin : parse_weight_channel
            for (w = 0; w < FILTER_SIZE * FILTER_SIZE; w = w + 1) begin : parse_weight_element
                always @(*) begin
                    weight_array[d][w] = weight[
                        (d * FILTER_SIZE * FILTER_SIZE + w) * DATA_BITS +: DATA_BITS
                    ];
                end
            end
        end
    endgenerate

    genvar k;
    generate
        for (k = 0; k < CHANNEL_LEN; k = k + 1) begin : compute_conv
            wire signed [DATA_BITS*2 - 1:0] prod [0:FILTER_SIZE * FILTER_SIZE - 1];
            wire signed [DATA_BITS*2 - 1:0] sum_prod [0:FILTER_SIZE * FILTER_SIZE - 1];

            for (i = 0; i < FILTER_SIZE * FILTER_SIZE; i = i + 1) begin : compute_products
                assign prod[i] = (exp_data[i] * weight_array[k][i]);
            end

            assign sum_prod[0] = prod[0];

            for (i = 1; i < FILTER_SIZE * FILTER_SIZE; i = i + 1) begin : compute_sums
                assign sum_prod[i] = sum_prod[i - 1] + prod[i];
            end

            assign calc_out[k * DATA_BITS +: DATA_BITS] = sum_prod[FILTER_SIZE * FILTER_SIZE - 1]>>>SHIFT_NUM;
        end
    endgenerate

    genvar e;
    generate
        for (e = 0; e < CHANNEL_LEN; e = e + 1) begin : resuult
            assign exp_bias[e] = $signed(bias[e* DATA_BITS+:DATA_BITS]);
            assign data_out[e*(DATA_BITS)+:DATA_BITS] = $signed(calc_out[e*DATA_BITS+:DATA_BITS]) + $signed(exp_bias[e]);

        end
    endgenerate

    assign valid = in_val;
    
    endmodule

