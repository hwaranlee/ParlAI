clear all; clc; close all;

root_dir='Y:\convai\ParlAI-v2\exp-opensub';
folder_name = 'exp-emb300-hs1024-lr0.0001-a_general';
filename = fullfile(root_dir, folder_name, sprintf('%s.log', folder_name));
% filename = fullfile(folder_name, sprintf('%s-2.log', folder_name));

fp = fopen(filename,'r');
tline = fgetl(fp);
loss = zeros(1000000,4);
loss_valid = zeros(1000000,3);
lrate_decay = zeros(20,4);

count = 0;
count_valid = 0;
count_lrdecay = 0;

loss_prev = 0;

while ischar(tline)
    
    parsed = strsplit(tline, {':', ',', ' ', '}'}, 'CollapseDelimiters',true);
    if (size(parsed,2) == 20) %18
        % training 
        count = count+1;
        loss(count,1) = str2double(parsed{1,10}); % updates
        loss(count,2) = str2double(parsed{1,17}); % ndata
        loss(count,3) = str2double(parsed{1,13}); % nll
        loss(count,4) = str2double(parsed{1,15}); % ppl
    elseif (size(parsed,2) == 15)
        % valid
        count_valid = count_valid + 1;
        loss_valid(count_valid,1) = loss(count,1); % updates
        loss_valid(count_valid,2) = str2double(parsed{1,8}); % nll       
        loss_valid(count_valid,3) = str2double(parsed{1,10}); % ppl
    elseif (size(parsed,2) == 8) & (strfind(tline, 'Decrease') == 27)
        % learing rate decy
        count_lrdecay = count_lrdecay + 1;
        lrate_decay(count_lrdecay, 1) = loss(count,1);
        lrate_decay(count_lrdecay, 2) = loss_valid(count_valid,2);% nll       
        lrate_decay(count_lrdecay, 3) = loss_valid(count_valid,3);% ppl
        lr = strsplit(parsed{1,9}, ']');
        lrate_decay(count_lrdecay, 4) = str2double(lr{1,1});% learning rate        
    end
    tline = fgetl(fp);
end

loss = loss(1:count,:);
loss_valid = loss_valid(1:count_valid,:);
lrate_decay = lrate_decay(1:count_lrdecay,:);

fclose(fp);

min(loss_valid(:,3))

iter_per_epoch =344;
epoch = floor(count/iter_per_epoch);
loss_avg = zeros(epoch+1,2);
loss_avg(:,1) = [1:epoch+1]'*34400;
for i=1:epoch*iter_per_epoch
   ep = floor(i/iter_per_epoch);
   loss_avg(ep+1,2) = loss(i,3) + loss_avg(ep+1,2);
end
loss_avg(:,2) = loss_avg(:,2) ./iter_per_epoch;


i=0;
lrdecay=0;
while (i < size(loss_valid,1))
   if isnan(loss_valid(i+1,2))
       loss_valid(i+1,:) = [];
       lrdecay = lrdecay + 1;
   else
    i=i+1;
   end   
end

epoch = size(loss_valid,1)
summary = [loss_avg(epoch,2), exp(loss_avg(epoch,2)), loss_valid(epoch, 2), loss_valid(epoch, 3)]
fprintf('epoch %d : ldecay %d \n', epoch, lrdecay);
fprintf('%s\n', folder_name);

%% Figure

figure(1);
set(gcf,'Color',[1 1 1])

subplot(1,2,1);
hold on;
plot(loss(:,1), loss(:,3), 'linewidth', 1);
plot(loss_avg(:,1), loss_avg(:,2), 'linewidth', 2);

plot(loss_valid(:,1), loss_valid(:,2), 'linewidth', 2);
scatter(lrate_decay(:,1), lrate_decay(:,2), '*');

hold off;
legend('Train', 'Train', 'Valid','location', 'northeast');
set(gca, 'ylim', [0, 7]);
set(gca, 'xtick', 0:34400:size(loss_valid,1)*34400, 'xticklabel', 0:size(loss_valid,1) , 'xlim', [0,loss(end,1)]);
xlabel('Epoch'); ylabel('NLL'); 

subplot(1,2,2);
hold on;
plot(loss(:,1), exp(loss(:,3)), 'linewidth', 1);
plot(loss_avg(:,1), exp(loss_avg(:,2)), 'linewidth', 2);

plot(loss_valid(:,1), exp(loss_valid(:,2)), 'linewidth', 2);
scatter(lrate_decay(:,1), lrate_decay(:,3), '*');

hold off;
legend('Train', 'Train', 'Valid','location', 'northeast');
set(gca, 'ylim', [0, 300]);
set(gca, 'xtick', 0:34400:size(loss_valid,1)*34400, 'xticklabel', 0:size(loss_valid,1) , 'xlim', [0,loss(end,1)]);
xlabel('Epoch'); ylabel('PPL'); 

saveas(gca, fullfile(root_dir, folder_name, sprintf('%s.fig', folder_name)), 'fig');


