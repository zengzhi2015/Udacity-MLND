clear
clc
%%
store_file = 'cls_test_32x32.mat';
load('cls_test_32x32_old.mat')
%%
X_cls = X;
y_cls = uint8(mod(y,10))';
%%
save(store_file,'X_cls','y_cls','-v7.3');