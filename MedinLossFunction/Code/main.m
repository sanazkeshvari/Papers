load fisheriris.mat


for iter=1:100
    [center, obj_fcn] = lncosh(meas(:,1));
    
    obj_fcn=smooth(obj_fcn);    obj_fcn=smooth(obj_fcn);

    figure(1);plot(obj_fcn)
    L=length(obj_fcn);
    Z=randperm(L);
    %  k1=round(L/3);k2=2*k1;
    %  k2=10;k1=5;
    for i=1:L-1
        k1=min(Z(i),Z(i+1));k2=max(Z(i),Z(i+1));
        LearningRate(i)=(k2-k1)/(log2(obj_fcn(k1)/obj_fcn(k2)));
        if isnan(LearningRate(i))
            LearningRate(i)
            k2
            k1
            obj_fcn(k1)
            obj_fcn(k2)
            
        end
        
    end
    LR(iter)=median(LearningRate);
end
LR=LR(find(LR>0));
median(LR)