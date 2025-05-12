%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取数据
res = xlsread('System.xlsx');

%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_dim = size(res, 2) - 1;               % 特征维度
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.7;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 1;                        % 标志位为1，打开混淆矩阵（要求2018版本及以上）

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
end

%%  数据转置
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  得到训练集和测试样本个数
M = size(P_train, 2);
N = size(P_test , 2);

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

%%  数据平铺
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致
P_train =  double(reshape(P_train, num_dim, 1, 1, M));
P_test  =  double(reshape(P_test , num_dim, 1, 1, N));

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  创建模型
layers = layerGraph();

layers = addLayers(layers, [
    sequenceInputLayer(num_dim, "Name", "input")                          % 建立输入层
    gruLayer(5, 'OutputMode', 'last', "Name", "gru1")                % GRU层
    concatenationLayer(1, 2, "Name", "cat")                          % 拼接层
    selfAttentionLayer(num_dim,num_dim,"Name","selfattention")       % 多头自注意力机制
    fullyConnectedLayer(num_class, "Name", 'fc')                     % 全连接层
    softmaxLayer("Name", "soft")                                     % 分类层
    classificationLayer("Name", "class")]);

layers = addLayers(layers, [
    FlipLayer("flip")                                          % 反转层
    gruLayer(5, 'OutputMode', 'last', "Name", "gru2" )] );    % 反向GRU

layers = connectLayers(layers, "input", "flip");
layers = connectLayers(layers, "gru2", "cat/in2");

%%  参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 500, ...                             % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.001, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', 400, ...                   % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.2, ...                    % 学习率调整因子
    'L2Regularization', 0.001, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'training-progress');                    % 画出曲线

%%  训练
[net, Loss] = trainNetwork(p_train, t_train, layers, options);

%%  预测
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test );

%%  反归一化
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  绘制网络结构
analyzeNetwork(net)

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', 'BiGRU-Multihead-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', 'BiGRU-Multihead-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
grid

%%  混淆矩阵
if flag_conusion == 1

    figure
    cm = confusionchart(T_train, T_sim1);
    cm.Title = 'Confusion Matrix for Train Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    
    figure
    cm = confusionchart(T_test, T_sim2);
    cm.Title = 'Confusion Matrix for Test Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end

%%  损失函数曲线
figure
subplot(2, 1, 1)
plot(1 : length(Loss.TrainingAccuracy), Loss.TrainingAccuracy, 'r-', 'LineWidth', 1)
xlabel('迭代次数')
ylabel('准确率')
legend('训练集正确率')
title ('训练集正确率曲线')
grid
set(gcf,'color','w')

subplot(2, 1, 2)
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, 'b-', 'LineWidth', 1)
xlabel('迭代次数')
ylabel('损失函数')
legend('训练集损失值')
title ('训练集损失函数曲线')
grid
set(gcf,'color','w')

%% 六边形图
[Metrics]=polygonareametric(T_test, T_sim2);

%% ROC曲线
[TPR_rf,FPR_rf,TRF_rf,AUCRF_rf]=perfcurve(T_test, T_sim2,1);
figure
plot(FPR_rf,TPR_rf,'r--','linewidth',1.5)
xlabel('假正类率FPR');ylabel('真正类率TPR')
hold on 
line([0 1],[0,1],'linewidth',1,'color','b'); 
hold on 
axis([0 1 0 1]); 
AUC_rf =trapz(FPR_rf,TPR_rf); 
string={'测试集ROC曲线结果'}; 
title(string) 
hold on 
text1=strcat('AUC=',num2str(AUC_rf)); 
legend(text1);
