clear all; close all; clc;

load('GPR.mat');
% Y is the output of the lift coefficient in phase with velocity (Clv).
% X is the input (normalized).
% X(:,1) is the reduced frequency (fr in [0.08, 0.35]) 
% X(:,2) is the non-dimensional CF amplitude (AyD in [0.05, 1.35]) 
%二维在所有点上做预测
%% Here is for the grid search
Xplot = linspace(0,4,100);%0-1之间每隔0.01取点，一维数组
% Here XNew is simply a vector from a 100*100 matrix, allows us to do grid
% search for the maximum of the standard deviation. Be aware for high
% dimensional input, this is not a good idea and optimaization method
% should be applied, such as fmincon(...) in the matlab.
XNew = zeros(100*100,2);%100*100矩阵
for jj=1:length(Xplot)
    bInd = (jj-1)*100+1;
    eInd = bInd +99;
    XNew(bInd:eInd,:) = [Xplot' ones(length(Xplot),1)*Xplot(jj)];%x,y=0-1之间的间隔0.01的网格
end
x=KC5(:,4:5);%归一化

y=KC5(:,1);
y1=KC5(:,2);
y2=KC5(:,3);
%KC=5_转换坐标

y=KC5(:,1).*(x(:,1))./(1+x(:,1)).*(1+x(:,2))./(x(:,2)+0.2).*(x(:,1))./(1+x(:,1)).*(1+x(:,2))./(x(:,2)+0.2);
y1=KC5(:,2).*(x(:,1))./(1+x(:,1)).*(1+x(:,2))./(x(:,2)+0.2).*(x(:,1))./(1+x(:,1)).*(1+x(:,2))./(x(:,2)+0.2);
y2=KC5(:,3).*(x(:,1))./(1+x(:,1)).*(1+x(:,2))./(x(:,2)+0.2).*(x(:,1))./(1+x(:,1)).*(1+x(:,2))./(x(:,2)+0.2);
x=KC5(:,4:5);%归一化

%x(:,2)=mapminmax(x(:,2), 0, 1);
%三维在所有点上做预测
% 定义三个维度的范围，这里以0到1为例
Xplot = linspace(0, 4, 100); % 第一维0-4bi1
Yplot = linspace(0, 4, 100); % 第二维0-4bi2
Zplot = linspace(0, 10, 100); % 第三维0-10KC

% 初始化三维网格数组，大小为 100*100*100，每个点有3个坐标
XNew2 = zeros(100*100*100, 3);

% 使用三个嵌套循环填充三维网格数组
% for kk = 1:length(Zplot)
%     for jj = 1:length(Xplot)
%         bInd = (kk - 1) * 10000 + (jj - 1) * 100 + 1; % 计算起始索引
%         eInd = bInd + 99; % 计算结束索引
%         
%         % 填充当前层的 x 和 y 坐标
%         XNew2(bInd:eInd, 1) = Xplot';
%         XNew2(bInd:eInd, 2) = Xplot(jj) * ones(100, 1);
%         
%         % 由于 z 坐标是当前层的常数，可以直接赋值
%         XNew2(bInd:eInd, 3) = Zplot(kk) * ones(100, 1);
%     end
% end%根据z增大,x增大，y增大，来排
% [x,y,z]=meshgrid(Xplot,Yplot,Zplot);
%  XNew3 = [x(:), y(:), z(:)];%这个是根据z增大,y增大，x增大来排的%使用XNew3而不是XNew2



%XFlattened = x5(:)';
%XMapped = mapminmax(XFlattened, 0, 1); % 在最大最小值之间全局归一化
%XMappedData = reshape(XMapped, size(x5));
XMappedData (:,1)=(ladingchaolifang3_10(:,5));%ladingchaolifang是load的数据4列是KC.5列是bi1,6列是bi2,1列是Cd，2列是
XMappedData (:,2)=(ladingchaolifang3_10(:,6));%没有归一化到0-1
XMappedData (:,3)=(ladingchaolifang3_10(:,4));%没有归一化到0-1
YMappedData=ladingchaolifang3_10(:,1);%Cd
YMappedData1=ladingchaolifang3_10(:,2);%Cm
%YMappedData2=ladingchaoladingchaolifanifang(:,3)./XMappedData (:,3)./XMappedData (:,3);%shang
YMappedData2=ladingchaolifang3_10(:,3);%E
%% Initial Sampling (has to be larger than the input dimension) 
% we select initially 6 samples, this can be chosen by latin hypercube
% method. In matlab: lhsdesign(...)
%x = XMappedData (1:300,:);%y = YMappedData(1:300);%修改1:？，这表示初始选取的点，
% x = XMappedData;
% y = YMappedData;%因为这里无迭代，因而选全部点
% %% Sequential Experimentation
% y1 = YMappedData1;
% y2 = YMappedData2;



% % kcyuce和 x 都是行向量化的，即每一行是一个单独的元素
% % 使用 setdiff 函数找到 kcyuce 中但不在 x 中的行
% x3 = setdiff(kcyuce, x, 'rows');
% 
% % 假设 y 是一个 100x6 的变量，x 是一个 20x3 的变量
% % 找出 x 在 y 中匹配的行索引
% % ismember 检查 x3 中的每一行是否存在于 kcyuce 的前三列中
% [~, idx] = ismember(x3, kcyuce(:, 1:3), 'rows');
% 
% % 根据行索引提取 kcyuce 中对应的后三列
% % 我们使用 idx 来索引 y 的后三列
% yCorresponding = KC1357(idx, 1:3);
for j = 1
    % please check Matlab manual for basis and kernel functions
    gprMdl = fitrgp(x,y,...
        'Basis','linear',...
        'FitMethod','exact',...
        'PredictMethod','exact',...
        'KernelFunction','ardmatern32');
    gprMdl1= fitrgp(x,y1,...
        'Basis','linear',...
        'FitMethod','exact',...
        'PredictMethod','exact',...
        'KernelFunction','ardmatern32');
    gprMdl2 = fitrgp(x,y2,...
        'Basis','linear',...
        'FitMethod','exact',...
        'PredictMethod','exact',...
        'KernelFunction','ardmatern32');
    % we acquire the mean (yprea) and standard deviation (yst) from the
    % learned GPR model with an input of XNew.
%     [ypred,yst] = predict(gprMdl,  kcyuce_KC357);%得到预测值与标准差
%     [ypred1,yst1] = predict(gprMdl1,  kcyuce_KC357);%得到预测值与标准差
%     [ypred2,yst2] = predict(gprMdl2,  kcyuce_KC357);%得到预测值与标准差
    
        % Via a grid searching method, wefind next experiment input and 
        % quantify the maximum of the STD at each iteration. For 
        % high-dimensional inputs, other searching methods may be applied, 
        % such as fmincon(...) in Matlab.
       %取得该点的位置和STD，1代表取STD最大的一个点（看要取几个点酌情改）
%        [a1,Ind1] = maxk(ypred,1);
%        [a3,Ind3] = maxk(ypred1,1);
%        [a5,Ind5] = maxk(ypred2,1);
        % the maximum of the STD at each iteration, can be used as one of 
        % the convergence rules
%         sigma(j) = a(1,1);
%     for kk = 1
%         % the next experiment input
%         xNext = XNew(Ind(kk,1),:); 
%         yNext=ypred(Ind(kk,1),:);
%         x = [x; xNext];
%         y a= [y; yNext];%根据最大标准差选择下一个点
%        
%     end
    
%     xNext = XMappedData(42+j,:); yNext =YMappedData(42+j,:);%记得修改42，42代表初始输入的数据点,我没有循环就不需要再加了，但是加不加对结果没影响
%     
%     x = [x; xNext];
%     y = [y; yNext];
    %GPRMDL{j,:} = gprMdl;
    GPRMDL{1,:}=gprMdl;GPRMDL{2,:}=gprMdl1;GPRMDL{3,:}=gprMdl2;


end
%clearvars -except GPRMDL 
%clearvars -except GPRMDL yst2 yst1 yst ypred2 ypred1 ypred XNew3 Ind x3 X1
%save('GPRCLVResult2.mat', 'GPRMDL','XNew','a',"a1","a2","a3","a4","a5","Ind","Ind1","Ind2","Ind3","Ind4","Ind5");%在kcyuce时不应该保存
save('GPR1-transfer.mat', 'GPRMDL','XNew');
%save('GPRCLVResultladingchaolifang+teshujianjvbi270个点', 'GPRMDL','ypred','yst','ypred1','yst1',"ypred2","yst2","XNew3");