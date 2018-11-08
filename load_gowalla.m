% function load_gowalla()
rating_dataPath = 'data/gowalla/visited_spots.txt';
social_network_path = 'data/gowalla/user_network.txt';
item_network_path = 'data/gowalla/spot_location.txt';
 
%%
tic;
gowalla_data = importdata(rating_dataPath);
reading_line = inf;
num_of_line = length(gowalla_data);
 
R = sparse([]);
for i = 1:min(reading_line, num_of_line)
    tline = gowalla_data{i};
    user_id = extractBefore(tline, ' [');
    user_id = str2double(user_id);
    poi_id = extractAfter(tline, '[');
    poi_id = extractBefore(poi_id,  ']');
    poi_list = split(poi_id, ', ');
    poi_list = str2num(char(poi_list));
    R(poi_list, user_id) = 1;
end
mask = R;
time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);
 
%% Read and transefer the spot data to spot network
tic;
fid = fopen(item_network_path,'r');
C = textscan(fid, '%f %f %f\n', reading_line);
fclose(fid);
 
poi = C{1};
lon = C{2};
lat = C{3};
 
lat_max = max(lat);
lat_min = min(lat);
lon_max = max(lon);
lon_min = min(lon);
 
 
sample_portion = 0.01;
 
num_spot = length(poi);
num_base = round(num_spot * sample_portion);
 
rng('default')
base_id = datasample(1:num_spot, num_base,'Replace', false);
 
lat_base = lat(base_id);
lon_base = lon(base_id);
 
 
 
 
% poi(base_id)
 
threshold = 0.2; %% This value is adapted by Carl Yang In PACE 2017.
 
 
base = [lat_base, lon_base];
lat_lon = [lat, lon];
% distance_eculidean = pdist2(base, lat_lon); 
% edge_exist = (distance_eculidean <= threshold);
 
 
edge = zeros(10000000,2);
% item_matrix = sparse([]);
ii = 1;
for i = 1:size(base,1)
    distance_eculidean = pdist2(base(i,:), lat_lon);
    edge_exist = (distance_eculidean <= threshold);
    
    poi_base = poi(base_id(i));
    num_i = sum(edge_exist);
    
    if (ii + num_i - 1) > size(edge,1)
        edge = [edge; zeros(size(edge,1),2)];
    end
    edge(ii: ii + num_i -1,:) = [poi_base * ones(sum(edge_exist),1), poi(edge_exist)];
    ii = ii + num_i;
%     item_matrix(poi_base, poi(edge_exist(i,:))) = 1;
end
edge(ii:end, :) = [];
edge(edge(:,1) == edge(:,2), :) = [];
item_matrix = sparse(edge(:,1), edge(:,2), ones(size(edge,1),1));
time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);
 
 
% distance_threshold = 10000; % meter
% for i = 1:num_base
%     neighbor = [];
%     poi_base = poi(base_id(i));
%     for j = 1:num_spot
%         distt = Dist(lat_base(i), lon_base(i), lat(j), lon(j));
%         if distt <= distance_threshold
%             neighbor = [neighbor poi(j)];
%         end
%     end
%     item_matrix(poi_base, neighbor) = 1;
%   item_matrix(neighbor, poi_base) = 1;
% end
 
 
 
%% Read Social network.
 
tic;
fid = fopen(social_network_path,'r');
C = textscan(fid, '%f %f %*f\n', reading_line);
fclose(fid);
 
truster = C{1};
trustee = C{2};
 
social_matrix = sparse(truster,trustee, ones(length(truster),1));
 
time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);
 
%% Data saving
 

%

num_user = min(length(social_matrix), size(R,2));
num_item = min(length(item_matrix), size(R,1));

R(num_item + 1 :end , :) = [];
R(:, num_user + 1 : end) = [];
social_matrix(num_user + 1 : end, :) = [];
social_matrix(:, num_user + 1 : end) = [];
item_matrix(num_item + 1 : end, :) = [];
item_matrix(:, num_item + 1 : end) = [];


if numel(item_matrix) < num_item * num_item
    item_matrix(num_item, num_item) = 0;
end
if numel(social_matrix) < num_user * num_user
    social_matrix(num_user, num_user) = 0;
end
if numel(R) < num_user * num_item
    R(num_item, num_user) = 0;
end
%

save('data/gowalla.mat', 'social_matrix', 'R', 'mask','item_matrix', 'map');
fprintf('Data saved successully!\n');

%% Obtain small dataset

user_threshold = 200;
item_threshold = 100;


reserved_user_ID = sum(R > 0,1) >= user_threshold;
reserved_item_ID = sum(R > 0,2) >= item_threshold;

fprintf("reserved user number: [%d]\n", full(sum(reserved_user_ID)));
fprintf("reserved item number: [%d]\n", full(sum(reserved_item_ID)));


social_matrix = social_matrix(reserved_user_ID, reserved_user_ID);
item_matrix = item_matrix(reserved_item_ID, reserved_item_ID);
R = R(reserved_item_ID, reserved_user_ID);
mask = mask(reserved_item_ID, reserved_user_ID);

save('data/gowalla_small.mat', 'social_matrix', 'R', 'mask','item_matrix');
fprintf('Small Data saved successully!\n');

%%
function distt = Dist(lat1, lon1, lat2, lon2)
%     lon1 = 104.07075;
%     lat1 = 30.664283;
%     lon2 = 103.93507;
%     lat2 = 30.760159;
    
    distt = distance(lat1,lon1,lat2,lon2);
    distt = distt*6371*1000*2*pi/360;
end
