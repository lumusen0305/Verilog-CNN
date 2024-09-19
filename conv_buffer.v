module conv_buffer #(parameter WIDTH = 28, HEIGHT = 28, DATA_BITS = 8,FILTER_SIZE = 5)(
input clk,
input in_val,
input rst_n,
input [DATA_BITS - 1:0] data_in,
output [(FILTER_SIZE*FILTER_SIZE-1)*DATA_BITS - 1:0] data_out,                                                       
output reg valid
);
//  localparam FILTER_SIZE = 5;
parameter READ = 1'b0;
parameter CAL  = 1'b1;
reg [DATA_BITS*WIDTH *(FILTER_SIZE-1)-1:0] buffer;
reg [DATA_BITS - 1:0] buf_idx,buf_idx_r;
reg [DATA_BITS*FILTER_SIZE-1:0] windows;
reg cur_state,nxt_state;
reg valid_r;
wire [DATA_BITS - 1:0]  buf_index = (buf_idx==0)?WIDTH-FILTER_SIZE:buf_idx-(FILTER_SIZE);

wire [DATA_BITS - 1:0] data_out_array [0:FILTER_SIZE*FILTER_SIZE];

genvar row, col;
generate
    for (row = 0; row < (FILTER_SIZE-1); row = row + 1) begin : gen_row
        for (col = 0; col < FILTER_SIZE; col = col + 1) begin : gen_col
            localparam integer i = row * 5 + col;
            assign data_out_array[i] = buffer[((buf_index + col) + WIDTH * row) * DATA_BITS +: DATA_BITS];
        end
    end
endgenerate
genvar i;
generate
    for (i = FILTER_SIZE*(FILTER_SIZE-1); i < FILTER_SIZE*FILTER_SIZE; i = i + 1) begin : gen_windows
        assign data_out_array[i] = windows[(i - FILTER_SIZE*(FILTER_SIZE-1)) * DATA_BITS +: DATA_BITS];
    end
endgenerate

genvar a;
generate
    for (a = 0; a < 25; a = a + 1) begin : gen_data_out_concat
        assign data_out[(a) * DATA_BITS +: DATA_BITS] = data_out_array[a];
    end
endgenerate

// wire  [DATA_BITS - 1:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4,
// data_out_5, data_out_6, data_out_7, data_out_8, data_out_9,
// data_out_10, data_out_11, data_out_12, data_out_13, data_out_14,
// data_out_15, data_out_16, data_out_17, data_out_18, data_out_19,
// data_out_20, data_out_21, data_out_22, data_out_23, data_out_24;
// assign data_out_0  =  buffer[ (buf_index  ) * DATA_BITS +: DATA_BITS];
// assign data_out_1  =  buffer[ (buf_index+1) * DATA_BITS +: DATA_BITS];
// assign data_out_2  =  buffer[ (buf_index+2) * DATA_BITS +: DATA_BITS];
// assign data_out_3  =  buffer[ (buf_index+3) * DATA_BITS +: DATA_BITS]; 
// assign data_out_4  =  buffer[ (buf_index+4) * DATA_BITS +: DATA_BITS];
// assign data_out_5  =  buffer[((buf_index  ) + WIDTH  )* DATA_BITS +: DATA_BITS];
// assign data_out_6  =  buffer[((buf_index+1) + WIDTH  )* DATA_BITS +: DATA_BITS];
// assign data_out_7  =  buffer[((buf_index+2) + WIDTH  )* DATA_BITS +: DATA_BITS];
// assign data_out_8  =  buffer[((buf_index+3) + WIDTH  )* DATA_BITS +: DATA_BITS];
// assign data_out_9  =  buffer[((buf_index+4) + WIDTH  )* DATA_BITS +: DATA_BITS];
// assign data_out_10 =  buffer[((buf_index  ) + WIDTH*2)* DATA_BITS +: DATA_BITS];
// assign data_out_11 =  buffer[((buf_index+1) + WIDTH*2)* DATA_BITS +: DATA_BITS];
// assign data_out_12 =  buffer[((buf_index+2) + WIDTH*2)    * DATA_BITS +: DATA_BITS];
// assign data_out_13 =  buffer[((buf_index+3) + WIDTH*2)* DATA_BITS +: DATA_BITS];
// assign data_out_14 =  buffer[((buf_index+4) + WIDTH*2)* DATA_BITS +: DATA_BITS];
// assign data_out_15 =  buffer[((buf_index  ) + WIDTH*3)* DATA_BITS +: DATA_BITS];
// assign data_out_16 =  buffer[((buf_index+1) + WIDTH*3)* DATA_BITS +: DATA_BITS];
// assign data_out_17 =  buffer[((buf_index+2) + WIDTH*3)* DATA_BITS +: DATA_BITS];
// assign data_out_18 =  buffer[((buf_index+3) + WIDTH*3)* DATA_BITS +: DATA_BITS];
// assign data_out_19 =  buffer[((buf_index+4) + WIDTH*3)* DATA_BITS +: DATA_BITS];
// assign data_out_20 = windows[0 * DATA_BITS +: DATA_BITS];
// assign data_out_21 = windows[1 * DATA_BITS +: DATA_BITS];
// assign data_out_22 = windows[2 * DATA_BITS +: DATA_BITS];
// assign data_out_23 = windows[3 * DATA_BITS +: DATA_BITS];
// assign data_out_24 = windows[4 * DATA_BITS +: DATA_BITS];
// assign data_out = {
// data_out_0 ,data_out_1 ,data_out_2 ,data_out_3 ,data_out_4 ,data_out_5 ,data_out_6 ,data_out_7 ,data_out_8 ,data_out_9 ,data_out_10,data_out_11,data_out_12,data_out_13,data_out_14,data_out_15,data_out_16,data_out_17,data_out_18,data_out_19,data_out_20,data_out_21,data_out_22,data_out_23,data_out_24
// };

always @(posedge clk or negedge rst_n) begin
if(~rst_n) begin
        buf_idx <= 0;
        cur_state<=0;
        buffer<=0;
        valid<=0;
end
else begin
        valid <= valid_r;
        buf_idx <= buf_idx_r;
        cur_state<=nxt_state;
        if(cur_state==READ) buffer[buf_idx*DATA_BITS+:DATA_BITS]<=data_in;
        else begin
            windows<={data_in,windows[DATA_BITS*FILTER_SIZE-1:DATA_BITS]};
            if (buf_idx==0 && valid     ) begin
                // buffer<={buffer[DATA_BITS*WIDTH-1:DATA_BITS*FILTER_SIZE],windows,buffer[DATA_BITS*WIDTH *(FILTER_SIZE-1)-1:DATA_BITS*WIDTH]};
                // buffer<={buffer[DATA_BITS*WIDTH-1:DATA_BITS*(FILTER_SIZE-1)],windows[DATA_BITS*FILTER_SIZE-1:DATA_BITS],buffer[DATA_BITS*WIDTH *(FILTER_SIZE-1)-1:DATA_BITS*WIDTH]};
                buffer<={buffer[DATA_BITS*WIDTH-1:0],buffer[DATA_BITS*WIDTH *(FILTER_SIZE-1)-1:DATA_BITS*WIDTH]};
            end
            else begin
                buffer[(buf_idx - FILTER_SIZE) * DATA_BITS +: DATA_BITS]<=windows[DATA_BITS-1:0];
            end
        end
end
end
always@(*)begin
    case (cur_state)
        READ:begin
            valid_r=0;
            if(in_val) begin
                if (buf_idx==    WIDTH * (FILTER_SIZE-1) - 1) begin
                    buf_idx_r=0;
                    nxt_state=CAL;
                end else begin
                    buf_idx_r=buf_idx+1;
                    nxt_state=cur_state;
                end
            end
            else begin 
                buf_idx_r=buf_idx; 
                nxt_state=cur_state;        
            end    
        end 
        CAL:begin
            nxt_state=cur_state;
            if (buf_idx==WIDTH -1 ) begin
                buf_idx_r=0;    
            end else begin
                buf_idx_r=buf_idx+1;
            end
            if (buf_idx>=FILTER_SIZE-1)valid_r=1;
            else valid_r=0;
        end 
    endcase
end
endmodule