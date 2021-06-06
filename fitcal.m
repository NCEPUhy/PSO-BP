function fitval = fitcal(pm,net,indim,hiddennum,outdim,D,Ptrain,Ttrain,minOutput,maxOutput)  
    [x,y,z]=size(pm);
    for i=1:x
        for j=1:hiddennum
            x2iw(j,:)=pm(i,((j-1)*indim+1):j*indim,z);
        end
        for k=1:outdim
            x2lw(k,:)=pm(i,(indim*hiddennum+1):(indim*hiddennum+hiddennum),z);
        end
        x2b=pm(i,((indim+1)*hiddennum+1):D,z);
        x2b1=x2b(1:hiddennum).';
        x2b2=x2b(hiddennum+1:hiddennum+outdim).';
        net.IW{1,1}=x2iw;
        net.LW{2,1}=x2lw;
        net.b{1}=x2b1;
        net.b{2}=x2b2;
        error=postmnmx(sim(net,Ptrain),minOutput,maxOutput)-postmnmx(Ttrain,minOutput,maxOutput);
        fitval(i,1,z)=mse(error);
    end