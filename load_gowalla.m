% function load_gowalla()
rating_dataPath = 'data/gowalla/visited_spots.txt';
social_network_path = 'data/gowalla/user_network.txt';
item_network_path = 'data/gowalla/spot_location.txt';

%%
tic;
gowalla_data = importdata(rating_dataPath);
reading_line = 50;
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

fclose(fid);

%% Read and transefer the spot data to spot network
tic;
fid = fopen(item_network_path,'r');
reading_line = 200;
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

threshold = 0.5; %% This value is adapted by Carl Yang In PACE 2017.


base = [lat_base, lon_base];
lat_lon = [lat, lon];
distance_eculidean = pdist2(base, lat_lon); 
edge_exist = (distance_eculidean <= threshold);


edge = [];
% item_matrix = sparse([]);
for i = 1:size(edge_exist,1)
    poi_base = poi(base_id(i));
    edge = [edge; [poi_base * ones(sum(edge_exist(i,:)),1), poi(edge_exist(i,:))]];
%     item_matrix(poi_base, poi(edge_exist(i,:))) = 1;
end
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
% 	item_matrix(neighbor, poi_base) = 1;
% end



%% Read Social network.

tic;
fid = fopen(social_network_path,'r');
reading_line = 500;
C = textscan(fid, '%f %f %*f\n', reading_line);
fclose(fid);

truster = C{1};
trustee = C{2};

social_matrix = sparse(truster,trustee, ones(length(truster),1));

time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);

%% Data saving

save('data/gowalla.mat', 'social_matrix', 'R', 'mask','item_matrix');
fprintf('Data saved successully!\n');

%%
function distt = Dist(lat1, lon1, lat2, lon2)
%     lon1 = 104.07075;
%     lat1 = 30.664283;
%     lon2 = 103.93507;
%     lat2 = 30.760159;
    
    distt = distance(lat1,lon1,lat2,lon2);
    distt = distt*6371*1000*2*pi/360;
end