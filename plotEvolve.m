clear all; close all; clc;
%load('GPR1-transfer.mat')%transfer
%load('GPRCLVResultladingchaolifang160个点.mat')%三维
load('GPRCLVResultKC=7 226个点1.mat')%二维
Cd = [[0:0.2:2.2]];%KC=5,变换后画图的等值面
Cm= [[0:0.2:2.2]];%%KC=5,变换后画图的等值面
Shang = [[0:0.5:5.5]];%%KC=5,变换后画图的等值面
Cd = [[0.6:0.2:2.6]];%KC=1
Cm = [[0:0.2:0.6],[0.7:0.1:1.3]];%%KC=1
Shang = [[0:0.1:1.]];%%KC=1
Cd = [[0.6:0.1:1.6]];%KC=3
Cm = [[0:0.2:0.6],[0.7:0.1:1.3]];%%KC=3
Shang = [[0.8:0.1:1.8]];%%KC=3
Cd = [[0.6:0.1:1.6]];%KC=5
Cm = [[0.4:0.1:1.4]];%%KC=5
Shang = [[0.8:0.2:1.4],[1.6:0.1:2.2]];%%KC=5
Cd = [[0.6:0.1:1.6]];%KC=7
Cm = [[0.8:0.05:1.3]];%%KC=7
Shang = [[0.:0.3:3]];%%KC=7
HE(1,:)=Cd;
HE(2,:)=Cm;
HE(3,:)=Shang;
Xplot = linspace(0,4,100);
bi1 = linspace(0,4,100);
bi2 = linspace(0,4,100);%归一化还原，真实范围
KC= linspace(0,10,100);
num = 1;%一个大图画三张图

% %二维-transfer

% for j = 1:3%无迭代
%     gprMdl = GPRMDL{j};
%     for i = 1:length(Xplot)
%         [ypred(:,i),yst(:,i)] = predict(gprMdl, [Xplot' ones(length(Xplot),1)*Xplot(i)]);
%     end
%     if j==1 || j==2
%         ypred=ypred/3;%取平均
%     end
%     a=HE(j,:);
%     % These are the plots of the prediction (mean)
%     figure(1)
%     subplot(1,3,num)
%     [C,h] = contour(bi1,bi2,ypred,a,'-k');
%     clabel(C,h,'FontSize',10,'Fontname', 'Times', 'LabelSpacing', 300);
%     hold on;
%     contour(bi1,bi2,ypred, [0 0], '-r', 'LineWidth',2);
%     set(gca,'FontSize',12);set(gca,'Fontname', 'Times')
%     ax = gca;ax.YTick = [0:0.5:4];
%     ax = gca;ax.XTick = [0.:0.5:4];
%     h = xlabel('$l_{01}$');set(h,'Interpreter','latex');
%     set(h,'FontSize',15);
%     h = ylabel('$l_{12}$');set(h,'Interpreter','latex');
%     set(h,'FontSize',15);
%     if j==1  
%     h = title( '$\hat{C_d}$' , 'Interpreter', 'latex');
%     elseif j==2
%         h = title('$\hat{C_m}$' , 'Interpreter', 'latex');
%     else
%         h = title('$\hat{E}$' , 'Interpreter', 'latex');
%     end
%     set(h,'FontSize',15);
%     hold off
%     
%     % This are the plots of the uncertainty (STD)
%     % The plots are not on the same color scale
%     figure(2)
%     subplot(1,3,num)
%     [C,h] = contourf(bi1,bi2,yst, '-k');
%     clabel(C,h,'FontSize',6,'Fontname', 'Times');
%     set(gca,'FontSize',6);set(gca,'Fontname', 'Times')
%     ax = gca;ax.YTick = [0 :0.125: 4];
%     ax = gca;ax.XTick = [0.:0.125:4];
%     h = xlabel('$l_{01}$');set(h,'Interpreter','latex');
%     set(h,'FontSize',10);
%     h = ylabel('$l_{12}$');set(h,'Interpreter','latex');
%     set(h,'FontSize',10);
%     if j==1
%     h = title(['Cd: KC=5' ]);
%     elseif j==2
%         h = title(['Cm: KC=5' ]);
%     else
%         h = title(['Enstrophy: KC=5']);
%     end
%     set(h,'FontSize',12);
%     hold off
%     
%     num = num+1;
% end
%二维
for j = 1:3%无迭代
    gprMdl = GPRMDL{j};
    for i = 1:length(Xplot)
        [ypred(:,i),yst(:,i)] = predict(gprMdl, [Xplot' ones(length(Xplot),1)*Xplot(i)]);
    end
    if j==1 || j==2
        ypred=ypred/3;%取平均
    end
    a=HE(j,:);
    % These are the plots of the prediction (mean)
    figure(1)
    subplot(1,3,num)
    [C,h] = contour(bi1,bi2,ypred,a,'-k');
    clabel(C,h,'FontSize',12,'Fontname', 'Times', 'LabelSpacing', 300);
    hold on;
    contour(bi1,bi2,ypred, [0 0], '-r', 'LineWidth',2);
    set(gca,'FontSize',12);set(gca,'Fontname', 'Times')
    ax = gca;ax.YTick = [0:0.5:4];
    ax = gca;ax.XTick = [0.:0.5:4];
    h = xlabel('$l_{01}$');set(h,'Interpreter','latex');
    set(h,'FontSize',15);
    h = ylabel('$l_{12}$');set(h,'Interpreter','latex');
    set(h,'FontSize',15);
    if j==1  
    h = title( '$C_d$' , 'Interpreter', 'latex');
    elseif j==2
        h = title('$C_m$' , 'Interpreter', 'latex');
    else
        h = title('$E$' , 'Interpreter', 'latex');
    end
    set(h,'FontSize',15);
    hold off
    
    % This are the plots of the uncertainty (STD)
    % The plots are not on the same color scale
    figure(2)
    subplot(1,3,num)
    [C,h] = contourf(bi1,bi2,yst, '-k');
    clabel(C,h,'FontSize',6,'Fontname', 'Times');
    set(gca,'FontSize',6);set(gca,'Fontname', 'Times')
    ax = gca;ax.YTick = [0 :0.125: 4];
    ax = gca;ax.XTick = [0.:0.125:4];
    h = xlabel('$l_{01}$');set(h,'Interpreter','latex');
    set(h,'FontSize',10);
    h = ylabel('$l_{12}$');set(h,'Interpreter','latex');
    set(h,'FontSize',10);
    if j==1
    h = title(['Cd: KC=5' ]);
    elseif j==2
        h = title(['Cm: KC=5' ]);
    else
        h = title(['Enstrophy: KC=5']);
    end
    set(h,'FontSize',12);
    hold off
    
    num = num+1;
end
% 
% 
% 
% %% 三维
% %     These are the plots of the prediction (mean)
% % 假设 ypred 是一个一维数组，我们将其重塑为三维网格格式
% % 定义等值面值
% % 定义等值面值
% % 
% % 假设 ypred 中的值代表每个点的高度或某种测量值
% % 使用 scatter3 绘制三维散点图
% % 假设 ypred 是一个一维数组，长度为 100*100*100
% % 我们需要将其重塑为一个三维矩阵，以匹配规则网格
% % 
% % % 创建规则网格
% [XGrid, YGrid, ZGrid] = meshgrid(linspace(0, 4, 100), linspace(0, 4, 100), linspace(0, 10, 100));
%  y1=3*XNew3(:,3)+2*XNew3(:,2)+XNew3(:,1);
% % 将 ypred 重塑为三维矩阵
% y1_matrix =reshape(y1, size(XGrid));
% ypred_matrix = reshape(ypred, size(XGrid))/3;%Cd平均
% ypred_matrix1 = reshape(ypred1, size(XGrid))/3;%Cm平均
% ypred_matrix2 = reshape(ypred2, size(XGrid));
% yst_matrix = reshape(yst, size(XGrid));
% yst_matrix1 = reshape(yst1, size(XGrid));
% yst_matrix2 = reshape(yst2, size(XGrid));
% % 定义等值面值
% % 您可以根据需要设置不同的等值面值
% [a,Ind] = maxk(yst_matrix,1);
% % 获取当前的 colormap
% cmap = jet(64); % 使用 64 个颜色级别的 jet 颜色映射
% 
% % 添加透明度通道，设置为半透明
% % 确保透明度矩阵与 cmap 具有相同的行数
% alpha_channel = 0.1 * ones(size(cmap, 1), 1); % 0.5 是透明度值
% cmap_alpha = [cmap, alpha_channel]; % 串联 RGB 和 Alpha 通道
% 
% % 使用 isosurface 绘制等值面
% figure(1);
% isoValues = 0.2:0.1:1.2;
% for  isoValue=isoValues
% isosurface(XGrid, YGrid, ZGrid, ypred_matrix, isoValue);
% end
%  %设置图形属性
% h = xlabel('$l_{01}$');set(h,'Interpreter','latex');
% h = ylabel('$l_{12}$');set(h,'Interpreter','latex');
% zlabel('KC');
% %title(['Isosurface Pred Plot for Cd at  ', num2str(isoValues)]);
% set(gca, 'FontSize', 20, 'Fontname', 'Times');
% view(3); % 设置为三维视图
% grid on; % 打开网格
% colormap jet; % 设置颜色映射
% caxis([0 1.5]);
% alpha(0.9);
% colorbar; % 显示颜色条
% 
% ax = gca; % 获取当前坐标轴
% set(ax, 'Color', 'none'); % 设置坐标轴背景色为透明
% hold off;
% 
% 
% figure(2);
% isoValues =0.6:0.07:1.4;
% for  isoValue=isoValues
%    isosurface(XGrid, YGrid, ZGrid, ypred_matrix1,isoValue);
% end
%  %设置图形属性
% h1 = xlabel('$l_{01}$');set(h1,'Interpreter','latex');
% h2 = ylabel('$l_{12}$');set(h2,'Interpreter','latex');
% zlabel('KC');
% %title(['Isosurface Pred Plot for Cm at  ', num2str(isoValues)]);
% set(gca, 'FontSize', 20, 'Fontname', 'Times');
% view(3); % 设置为三维视图
% grid on; % 打开网格
% colormap jet; % 设置颜色映射
% % 设置颜色映射的数据范围
% caxis([0 1.5]);
% colorbar; % 显示颜色条
% ax = gca; % 获取当前坐标轴
% set(ax, 'Color', 'none'); % 设置坐标轴背景色为透明
% alpha(0.9);
% 
% 
% figure(3);
% hold off;
% isoValues =0.:0.3:3;
% for  isoValue=isoValues
% isosurface(XGrid, YGrid, ZGrid, ypred_matrix2, isoValue);
% end
% % 设置图形属性
% h = xlabel('$l_{01}$');set(h,'Interpreter','latex');
% h = ylabel('$l_{12}$');set(h,'Interpreter','latex');
% zlabel('KC');
% %title(['Isosurface Plot for Enstrophy at  ', num2str(isoValues)]);
% set(gca, 'FontSize', 20, 'Fontname', 'Times');
% view(3); % 设置为三维视图
% grid on; % 打开网格
% colormap jet; % 设置颜色映射
% caxis([0 4]);
% colorbar; % 显示颜色条
% alpha(0.9);
% ax = gca; % 获取当前坐标轴
% set(ax, 'Color', 'none'); % 设置坐标轴背景色为透明
% hold off;
% 
% 
% figure(4);
% isoValues =5:5:10;
% for  isoValue=isoValues
% isosurface(XGrid, YGrid, ZGrid, y1_matrix, isoValue);
% end
% % 设置图形属性
% h = xlabel('$l_{01}$');set(h,'Interpreter','latex');
% h = ylabel('$l_{12}$');set(h,'Interpreter','latex');
% zlabel('KC');
% %title(['Isosurface Plot for jiashujv at  ', num2str(isoValues)]);
% set(gca, 'FontSize', 14, 'Fontname', 'Times');
% view(3); % 设置为三维视图
% grid on; % 打开网格
% colormap jet; % 设置颜色映射
% colorbar; % 显示颜色条


% %求Cd最大值随KC变化
% for jj=1:length(Xplot)
%     ii=(jj-1)*10000+1;
%     jjj=ii+9999;
%   [a,Ind] = maxk(ypred(ii:jjj),1);
%   sigma(jj) = a(1,1);
%   locbi2(jj)=mod (Ind(1,1)-1,100)/99*4;
%   locbi1(jj)=(mod (Ind(1,1)-1,10000)-mod (Ind(1,1)-1,100))/100/99*4;
% end
% figure(1)
% title('prediction of Cd max location with KC')% 创建左侧 y 轴
% yyaxis left;
% plot(KC,locbi2,'b');
% xlabel('KC');
% ylabel('bi2');
% yyaxis right;
% plot(KC,locbi1,'r');
% xlabel('KC');
% ylabel('bi1');
% % 设置x轴的显示范围为大于1
% xlim([1, max(KC)]);
% ylim([0, 4]);
% hold on;
% figure(2)
% for jj=1:length(Xplot)
%     ii=(jj-1)*10000+1;
%     jjj=ii+9999;
%   [a,Ind] = maxk(ypred1(ii:jjj),1);
%   sigma(jj) = a(1,1);
%   locbi2(jj)=mod (Ind(1,1)-1,100)/99*4;
%   locbi1(jj)=(mod (Ind(1,1)-1,10000)-mod (Ind(1,1)-1,100))/100/99*4;
% end
% 
% % plot(KC,loc);
% % title('prediction of Cm max bi2 with KC')
% % xlabel('KC');
% % ylabel('Cm');
% 
% title('prediction of Cm max location with KC')% 创建左侧 y 轴
% yyaxis left;
% plot(KC,locbi2,'b');
% xlabel('KC');
% ylabel('bi2');
% yyaxis right;
% plot(KC,locbi1,'r');
% xlabel('KC');
% ylabel('bi1');
% % 设置x轴的显示范围为大于1
% xlim([1, max(KC)]);
% ylim([0, 4]);
% hold on;
% 
% figure(3)
% for jj=1:length(Xplot)
%     ii=(jj-1)*10000+1;
%     jjj=ii+9999;
%   [a,Ind] = maxk(ypred2(ii:jjj),1);
%   sigma(jj) = a(1,1);
%   locbi2(jj)=mod (Ind(1,1)-1,100)/99*4;
%   locbi1(jj)=(mod (Ind(1,1)-1,10000)-mod (Ind(1,1)-1,100))/100/99*4;
% end
% % plot(KC,loc);
% % title('prediction of Enstrophy max bi2 with KC')
% % xlabel('KC');
% % ylabel('Enstrophy');
% 
% title('prediction of E max location with KC')% 创建左侧 y 轴
% yyaxis left;
% plot(KC,locbi2,'b');
% xlabel('KC');
% ylabel('bi2');
% yyaxis right;
% plot(KC,locbi1,'r');
% xlabel('KC');
% ylabel('bi1');
% % 设置x轴的显示范围为大于1
% xlim([1, max(KC)]);
% ylim([0, 4]);
% hold on;
% hold off;