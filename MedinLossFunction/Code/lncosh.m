function [center, obj_fcn_final] = lncosh(meas) 
    n=size(meas,1);
    iteration=100;
    gamma = 0.5;
    m=0;
    m1=0;
    L= length(meas);
    z= randperm(L,1);
    %z=52;
    center=meas(z,:);
    
    for i1=1:iteration
        obj_fcn=0;
        center_old = center;
        for i=1:n
            e=meas(i,:)-center;
            q=1+((e.^2))+(4*((e.^3))/6);
            w=q./(exp((e.^2))+1);
            m = m + meas(i,:)*w;
            if isinf(m)
                m
            end
            m1= m1 + w;
            obj_fcn= obj_fcn + log(cosh(e));
        end
        obj_fcn_final(i1)=obj_fcn;
        center = m/m1 ;
        if abs(center-center_old)<0.001
            break
        end
    end
end