module conv_buffer_tb;
    parameter WIDTH = 28;
    parameter HEIGHT = 28;
    parameter DATA_BITS = 8;
    parameter FILTER_SIZE = 5;
    parameter TOTAL_PIXELS = WIDTH * HEIGHT;

    reg clk;
    reg rst_n;
    reg in_val;
    reg [DATA_BITS - 1:0] data_in;
    wire [(FILTER_SIZE * FILTER_SIZE) * DATA_BITS - 1:0] data_out;
    wire valid;

    // Instantiate the conv_buffer module
    conv_buffer #(
        .WIDTH(WIDTH),
        .HEIGHT(HEIGHT),
        .DATA_BITS(DATA_BITS),
        .FILTER_SIZE(FILTER_SIZE)
    ) dut (
        .clk(clk),
        .in_val(in_val),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_out(data_out),
        .valid(valid)
    );

    // Testbench variables
    integer pixel_idx;
    integer i, j, k;
    reg match;
    reg [7:0] pixels [0:783];

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz clock

    // Initialize signals
    initial begin
        $readmemh("3_0.txt", pixels);
        clk <= 1'b0;
        rst_n <= 1'b1;
        #3
        rst_n <= 1'b0;
        #3
        rst_n <= 1'b1;

        // Start sending data
        @(negedge clk);
        in_val = 1;
        for (pixel_idx = 0; pixel_idx < TOTAL_PIXELS; pixel_idx = pixel_idx + 1) begin
            data_in = pixels[pixel_idx];
            @(negedge clk);
        end
        in_val = 0;

        // Wait for remaining outputs
        repeat(5) @(posedge clk);
        // Finish simulation
        $finish;
    end
    initial begin
      $fsdbDumpfile("wave.fsdb");
      $fsdbDumpvars(0, conv_buffer_tb);
    end
    // Monitor valid signal and check data_out
    integer i=0;
    integer window_row =0;
    integer window_col =0;
    integer row, col;
    // wire [32:0] img_idx=(window_row + row) * WIDTH + (window_col + col);
    reg [32:0] check_index=0;
    reg [DATA_BITS - 1:0] expected_window [0:FILTER_SIZE * FILTER_SIZE - 1];

    always @(posedge clk) begin
        if (valid) begin
            check_index<=check_index+1;
    //         // Extract expected 5x5 window based on current pixel index

    //         // Calculate the top-left pixel index of the window
            if(window_col==WIDTH-FILTER_SIZE)begin
                window_col<=0;
                window_row<=window_row+1;    
            end else begin
                window_col<=window_col+1;
            end

    //         // Extract expected window data
            for (row = 0; row < FILTER_SIZE; row = row + 1) begin
                for (col = 0; col < FILTER_SIZE; col = col + 1) begin
                    // integer img_idx = (window_row + row) * WIDTH + (window_col + col);
                    expected_window[row * FILTER_SIZE + col] = pixels[(window_row + row) * WIDTH + (window_col + col)];
                    $display("Expected pixels[%d]:  %h",(window_row + row) * WIDTH + (window_col + col),pixels[(window_row + row) * WIDTH + (window_col + col)]);
                end
            end

    //         // Compare expected window with data_out
            match = 1;
            for (k = 0; k < FILTER_SIZE * FILTER_SIZE; k = k + 1) begin
                if (data_out[(k) * DATA_BITS +: DATA_BITS] !== expected_window[k]) begin    
                    match = 0;
                    $display("[Mismatch:pixel[%d]] Expected expected_window[%d] %h, Got data_out[%d] %h",check_index,k , expected_window[k],k, data_out[(k) * DATA_BITS +: DATA_BITS]);
                end
            end

            if (match) begin
                $display("=========================================");
                $display("[PASS]: data_out matches expected window");
                $display("=========================================");
            end else begin
                $display("=========================================");
                $display("Mismatch detected at window %d", (window_row) * WIDTH + (window_col));
                $display("=========================================");
            end
        end
    end

endmodule