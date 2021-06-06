
% ��������Ⱥ�㷨�Ż�BP������
tic
clc
clear all

st=cputime;
%��������
load('zhangdiefu.mat');%�ǵ���
load('low.mat')%��
load('high.mat')%��
load('shoupan.mat')%���̣�Ԥ��Ŀ��ֵ��
load('jiaoyiliang.mat')%������
load('kaipan.mat')%kaipan
m1=0.8*size(low,1);
m2=m1+1;
m3=size(low,1);

%ѵ������
A1=jiaoyiliang(1:m1,1)';
A2=high(1:m1,1)';
A3=low(1:m1,1)';
A4=kaipan(1:m1,1)';
A5=zhangdiefu(1:m1,1)';
A6=shoupan(1:m1,1)';
%Ԥ������
A1a=shoupan(m2:m3,1)';
A2a=high(m2:m3,1)';
A3a=low(m2:m3,1)';
A4a=kaipan(m2:m3,1)';
A5a=zhangdiefu(m2:m3,1)';
A6a=shoupan(m2:m3,1)';

%����ѵ����������
Input=[A1;A2;A3;A4;A5];

%����ѵ���������
Output=A6;

% Ԥ����ѹ����[-1,+1]
global minOutput;
global maxOutput;
[Inputn,minInput,maxInput,Outputn,minOutput,maxOutput] = premnmx(Input,Output);
[Inputn,InputStr]=mapminmax(Input);
[Outputn,OutputStr]=mapminmax(Output);

%�������ݣ���20%Ϊ�������ݣ�
TestSamIn=[A1a;A2a;A3a;A4a;A5a];

%�������ݵ���ʵ������������̼۸�ĺ�20%���ݣ�

RealTestSamOut=A6a;

[TestSamInn,minTestSamIn,maxTestSamIn,RealTestSamOutn,minRealTestSamOut,maxRealTestSamOut] = premnmx(TestSamIn,RealTestSamOut);

% ѵ������
TrainSamIn=Input;
TrainSamOut=Output;
[TrainSamInn,minTrainSamIn,maxTrainSamIn,TrainSamOutn,minTrainSamOut,maxTrainSamOut] = premnmx(TrainSamIn,TrainSamOut);

%**********************************************************
% ѵ��������
global Ptrain;
Ptrain = TrainSamInn;
global Ttrain;
Ttrain = TrainSamOutn;
% ���Լ�����
Ptest = TestSamInn;

%**********************************************************
% BP�����������ʼ��
global indim;
indim=5;
global hiddennum;
hiddennum=11;
global outdim;
outdim=1;
%**********************************************************
%����Ⱥ�㷨������ʼ��
vmax=0.5; % �������ٶ�
minerr=0.0001; % �����Сֵ
wmax=0.90;
wmin=0.30;
global itmax; %�����������ֵ
itmax=300;


for iter=1:itmax
   
    W(iter)=wmax-((wmax-wmin)/(2^(iter/itmax)))*((iter/itmax)^2); % weight declining linearly
   
end 
cmax=2.1;
cmin=0.8;
c=cmax-(cmax-cmin)/(iter/itmax);
c1=c;
c2=c;
a=-1;  
b=1;
m=-1;
n=1;
global N; % ��������
N=100;
global D; % ���ӳ���
D=(indim+1)*hiddennum+(hiddennum+1)*outdim;
global fvrec;
MinFit=[];
BestFit=[];

% ����λ�ó�ʼ��
rand('state',sum(100*clock));
X=a+(b-a)*rand(N,D,1);
%Initialize velocities of particles
V=m+(n-m)*rand(N,D,1);

%**********************************************************
%Ŀ�꺯��ֵ��С
global net;
net=newff(minmax(Ptrain),[hiddennum,outdim],{'tansig','purelin'});
fitness=fitcal(X,net,indim,hiddennum,outdim,D,Ptrain,Ttrain,minOutput,maxOutput);
fvrec(:,1,1)=fitness(:,1,1);

[C,I]=min(fitness(:,1,1));
MinFit=[MinFit C];
BestFit=[BestFit C];
L(:,1,1)=fitness(:,1,1); %��¼ÿ�ε��������е�����ֵ
B(1,1,1)=C;  %��СĿ��ֵ��¼
gbest(1,:,1)=X(I,:,1);  %��СĿ��ֵʱȨֵ��ֵ��λ��
%********************************************************
%�������Ӹ�ֵ 
for p=1:N 
    G(p,:,1)=gbest(1,:,1);
end
for i=1:N;
    pbest(i,:,1)=X(i,:,1);
end
V(:,:,2)=W(1)*V(:,:,1)+c1*rand*(pbest(:,:,1)-X(:,:,1))+c2*rand*(G(:,:,1)-X(:,:,1));%�ٶȸ���
for ni=1:N
    for di=1:D
        if V(ni,di,2)>vmax
            V(ni,di,2)=vmax;
        elseif V(ni,di,2)<-vmax
            V(ni,di,2)=-vmax;
        else
            V(ni,di,2)=V(ni,di,2);
        end
    end
end
X(:,:,2)=X(:,:,1)+V(:,:,2);
%******************************************************
for j=2:itmax 
    
% ����λ�ø��� 
    fitness=fitcal(X,net,indim,hiddennum,outdim,D,Ptrain,Ttrain,minOutput,maxOutput);
    fvrec(:,1,j)=fitness(:,1,j);
   
    [C,I]=min(fitness(:,1,j));
    MinFit=[MinFit C];   
    BestFit=[BestFit min(MinFit)];
    L(:,1,j)=fitness(:,1,j);
    B(1,1,j)=C;
    gbest(1,:,j)=X(I,:,j);
    [C,I]=min(B(1,1,:));
    % ��ѡȨֵ��ֵ
    if B(1,1,j)<=C
        gbest(1,:,j)=gbest(1,:,j); 
    else
        gbest(1,:,j)=gbest(1,:,I);
    end 
    if C<=minerr, break, end
    %Ȩֵ��ֵ���� 
    if j>=itmax, break, end
    for p=1:N
         G(p,:,j)=gbest(1,:,j);
    end
    for i=1:N;
        [C,I]=min(L(i,1,:));
        if L(i,1,j)<=C
            pbest(i,:,j)=X(i,:,j);
        else
            pbest(i,:,j)=X(i,:,I);
        end
    end
    V(:,:,j+1)=W(j)*V(:,:,j)+c1*rand*(pbest(:,:,j)-X(:,:,j))+c2*rand*(G(:,:,j)-X(:,:,j));
    
    for ni=1:N
        for di=1:D
            if V(ni,di,j+1)>vmax
                V(ni,di,j+1)=vmax;
            elseif V(ni,di,j+1)<-vmax
                V(ni,di,j+1)=-vmax;
            else
                V(ni,di,j+1)=V(ni,di,j+1);
            end
        end
    end
    X(:,:,j+1)=X(:,:,j)+V(:,:,j+1);
end
disp('Iteration and Current Best Fitness')
disp(j)
disp(B(1,1,j))
disp('Global Best Fitness and Occurred Iteration')
[C,I]=min(B(1,1,:))
% �������
for t=1:hiddennum
    x2iw(t,:)=gbest(1,((t-1)*indim+1):t*indim,j);
end
for r=1:outdim
    x2lw(r,:)=gbest(1,(indim*hiddennum+1):(indim*hiddennum+hiddennum),j);
end
x2b=gbest(1,((indim+1)*hiddennum+1):D,j);
x2b1=x2b(1:hiddennum).';
x2b2=x2b(hiddennum+1:hiddennum+outdim).';
net.IW{1,1}=x2iw;
net.LW{2,1}=x2lw;
net.b{1}=x2b1;
net.b{2}=x2b2;

TrainSamOut=sim(net,Ptrain);
[b]=postmnmx(TrainSamOut,minTrainSamOut,maxTrainSamOut);
b
TestSamOut = sim(net,Ptest);
[a]=postmnmx(TestSamOut,minRealTestSamOut,maxRealTestSamOut);
ae=abs(a-RealTestSamOut)
mae=mean(ae)
re=(a-RealTestSamOut)./RealTestSamOut
mre=mean(abs(re))
a
re2=a-RealTestSamOut;
re4=RealTestSamOut-a;
Realerror=re4;
PSOBPoutput=a;
PSOBPerror=A6-b;
number=size(a,2);
PredictZD=[];%Ԥ�������ǵ�������
for numberZD=1:number-1
    numberZY=numberZD+1;
PredictZD(1,numberZD)=100*(a(1,numberZD)-a(1,numberZY))/a(1,numberZY);%����ǰ�ǵ������㹫ʽ
end
%plot(MaxFit,'k');
plot(log(BestFit),'--k','linewidth',2);
title('��Ӧ��');
xlabel('��������');
ylabel('fit');

figure 
grid
hold on
t1=1:size(a,2);

%Ԥ��ͼ��ǰ�Ÿ�����Ϊʵ�����ݣ�����������ΪԤ����������
plot(t1,a,'-',t1,RealTestSamOut,'--');
legend('����ǰ���̼�Ԥ������','���̼���ʵ����')
xlabel('ʱ��'); ylabel('Ԥ��ֵ');
title('����ǰPSO-BP����Ԥ�����ݶԱ�ͼ');
figure
grid
hold on
plot(ae,'k','linewidth',2);
title('�������');
xlabel('ʱ��');
ylabel('���');


figure
grid
hold on
plot(re,'k','linewidth',2);
title('������');
xlabel('ʱ��');
ylabel('���');

figure
grid
hold on
plot(re4,'k','linewidth',2);
title('����ǰ����в�ֵ');
xlabel('ʱ��');
ylabel('���');


et=cputime-st;

grid on;
RMSE=sqrt(sum((RealTestSamOut-a).^2)/length(a));%RMSE��������ģ�͵�Ԥ������
disp('����ǰ���Ԥ��ģ�͵ľ��������');
RMSE
save huangjinpredictdata a b re2 re RealTestSamOut PredictZD
toc



