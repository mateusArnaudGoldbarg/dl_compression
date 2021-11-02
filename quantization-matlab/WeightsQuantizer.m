function [WQS,BQS,WQ,BQ] = WeightsQuantizer(W,B,N)
    

     maxW = max(max(max(max(abs(W)))));
     maxB = max(max(max(max(abs(B)))));
     maxP = max([maxW maxB]);
    
     q =  maxP/(2^(N-1));
     
     WQ = round(W./q);
     BQ = round(B./q);
     
     WQS = WQ*q;
     BQS = BQ*q;

end
    




