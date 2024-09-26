module conv_buffer #(parameter WIDTH = 28, HEIGHT = 28, DATA_BITS = 8,FILTER_SIZE = 5)(
input clk,
input in_val,
input rst_n,
input [DATA_BITS - 1:0] data_in,
output [(FILTER_SIZE*FILTER_SIZE)*DATA_BITS - 1:0] data_out,                                                       
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

wire [DATA_BITS - 1:0] data_out_array [0:FILTER_SIZE*FILTER_SIZE-1];

wire  [DATA_BITS*WIDTH -1:0] buffer_list[(FILTER_SIZE-2):0];

genvar b;
generate
    for (b = 0; b < (FILTER_SIZE-1); b = b + 1) begin : filter_b
            assign buffer_list[b] = buffer[b * DATA_BITS*WIDTH +: DATA_BITS*WIDTH];
    end
endgenerate


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
    for (a = 0; a < FILTER_SIZE*FILTER_SIZE; a = a + 1) begin : gen_data_out_concat
        assign data_out[(a) * DATA_BITS +: DATA_BITS] = data_out_array[a];
    end
endgenerate


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
            if (in_val) begin
            windows<={data_in,windows[DATA_BITS*FILTER_SIZE-1:DATA_BITS]};
            if (buf_idx==0 && valid) begin
                buffer<={windows,buffer[(DATA_BITS*(WIDTH-FILTER_SIZE))-1:0],buffer[DATA_BITS*WIDTH *(FILTER_SIZE-1)-1:DATA_BITS*WIDTH]};
            end
            else if(buf_idx>(FILTER_SIZE-1))begin
                buffer[(buf_idx - FILTER_SIZE) * DATA_BITS +: DATA_BITS]<=windows[DATA_BITS-1:0];
            end
            end
        end
end
end
always@(*)begin
    case (cur_state)
        READ:begin
            valid_r=0;
            if(in_val) begin
                if (buf_idx==WIDTH * (FILTER_SIZE-1) - 1) begin
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
            if(in_val) begin
                nxt_state=cur_state;
                if (buf_idx==WIDTH -1 ) begin
                    buf_idx_r=0;    
                end else begin
                    buf_idx_r=buf_idx+1;
                end
                if (buf_idx>=FILTER_SIZE-1)valid_r=1;
                else valid_r=0;
            end 
            else begin
                buf_idx_r=buf_idx; 
                nxt_state=cur_state;
                valid_r=0;
            end
        end 
    endcase
end
endmodule


module conv_buffer_2w #(parameter WIDTH = 28, HEIGHT = 28, DATA_BITS = 8,FILTER_SIZE = 5)(
input clk,
input in_val,
input rst_n,
input [DATA_BITS - 1:0] data_in,
output [(FILTER_SIZE*FILTER_SIZE)*DATA_BITS - 1:0] data_out,                                                       
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

wire [DATA_BITS - 1:0] data_out_array [0:FILTER_SIZE*FILTER_SIZE-1];

wire  [DATA_BITS*WIDTH -1:0] buffer_list[(FILTER_SIZE-2):0];

genvar b;
reg flag,flag_r;

generate
    for (b = 0; b < (FILTER_SIZE-1); b = b + 1) begin : filter_b
            assign buffer_list[b] = buffer[b * DATA_BITS*WIDTH +: DATA_BITS*WIDTH];
    end
endgenerate


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
    for (a = 0; a < FILTER_SIZE*FILTER_SIZE; a = a + 1) begin : gen_data_out_concat
        assign data_out[(a) * DATA_BITS +: DATA_BITS] = data_out_array[a];
    end
endgenerate
reg stripe_cnt,stripe_cnt_r;
always @(posedge clk or negedge rst_n) begin
if(~rst_n) begin
        buf_idx <= 0;
        stripe_cnt<=0;
        cur_state<=0;
        buffer<=0;
        valid<=0;
        flag<=1;
end
else begin
        stripe_cnt<=stripe_cnt_r;
        flag<=flag_r;
        valid <= valid_r;
        buf_idx <= buf_idx_r;
        cur_state<=nxt_state;
        if(cur_state==READ) buffer[buf_idx*DATA_BITS+:DATA_BITS]<=data_in;
        else begin
            windows<={data_in,windows[DATA_BITS*FILTER_SIZE-1:DATA_BITS]};
            if (buf_idx==0 && valid) begin
                buffer<={windows,buffer[(DATA_BITS*(WIDTH-FILTER_SIZE))-1:0]};
            end
            else if(buf_idx>(FILTER_SIZE-1))begin
                buffer[(buf_idx - FILTER_SIZE) * DATA_BITS +: DATA_BITS]<=windows[DATA_BITS-1:0];
            end
        end
end
end
always@(*)begin
    case (cur_state)
        READ:begin
            flag_r=1;
            valid_r=0;
            if(in_val) begin
                if (buf_idx==WIDTH * (FILTER_SIZE-1) - 1) begin
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
            stripe_cnt_r=stripe_cnt;

        end 
        CAL:begin
            if(in_val) begin
                nxt_state=cur_state;
                if (buf_idx==WIDTH -1 ) begin
                    buf_idx_r=0;
                    flag_r=!flag;  
                end else begin
                    buf_idx_r=buf_idx+1;
                    flag_r=flag;
                end
                if (buf_idx>=FILTER_SIZE-1 && flag ) begin
                    if(!stripe_cnt)valid_r=1;
                    else valid_r=0;
                    stripe_cnt_r=stripe_cnt+1;
                end
                else begin
                    valid_r=0;
                    stripe_cnt_r=0;
                end
            end 
            else begin
                flag_r=flag;
                stripe_cnt_r=stripe_cnt;
                buf_idx_r=buf_idx; 
                nxt_state=cur_state;
                valid_r=0;
            end
        end
    endcase
end
endmodule