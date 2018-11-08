rating_dataPath = 'data/yelp_dataset/yelp_academic_dataset_review.json';
social_network_path = 'data/yelp_dataset/yelp_academic_dataset_user.json';
item_network_path = 'data/yelp_dataset/yelp_academic_dataset_business.json';
 
%% Prepare to read data
 
reading_line = 300;
partial = true;
 
if isinf(reading_line)
    partial = false;
    rating_data = importdata(rating_dataPath);
    social_network_data = importdata(social_network_path);
    item_network_data = importdata(item_network_path);
end
 
 
%% Read rating data
 
% 
% map_from_user_to_colID = containers.Map();
% if partial
%     map_from_colID_to_user = cell(1000000,1);
% else
%     map_from_colID_to_user = cell(length(social_network_data),1);
% end
% map_from_item_to_rowID = containers.Map();
% if partial
%     map_from_rowID_to_item = cell(1000000,1);
%     map_from_rowID_to_category = cell(1000000,1);
% else
%     map_from_rowID_to_item = cell(length(item_network_data),1);
%     map_from_rowID_to_category = cell(length(item_network_data),1);
% end
% 
% 
% 
% line = 0;
% colID = 1;
% rowID = 1;
% 
% if partial
%     item_user_list = zeros(10000,3);
% else
%     item_user_list = zeros(length(rating_data),3);
% end
% 
% if partial
%     fid = fopen(rating_dataPath, 'r');
% end
% 
% 
% tic;
% while true
%     line = line + 1;
%     if partial
%         if feof(fid) || line >= reading_line
%             break;
%         end
%         line_str = fgetl(fid);
%         line_data = jsondecode(line_str);
%     else
%         if line >= length(rating_data)
%             break;
%         end
%         line_str = rating_data{line};
%         line_data = jsondecode(line_str);
%     end   
%     item = line_data.business_id;
%     user = line_data.user_id;
%     star = line_data.stars;
% 
%     if ~isKey(map_from_user_to_colID, user)
%         map_from_colID_to_user{colID} = user;
%         map_from_user_to_colID(user) = colID;
%         col_this = colID;
%         colID = colID + 1;
%     else
%         col_this = map_from_user_to_colID(user);
%     end
% 
%     if ~isKey(map_from_item_to_rowID, item)
%         map_from_rowID_to_item{rowID} = item;
%         map_from_item_to_rowID(item) = rowID;
%         row_this = rowID;
%         rowID = rowID + 1;
%     else
%         row_this = map_from_item_to_rowID(item);
%     end
% 
%     item_user_list(line, :) = [row_this, col_this, star];
% end
% 
% if partial
%     fclose(fid);
% end
% 
% item_user_list(line+1:end,:) = [];
% map_from_colID_to_user(colID:end) = [];
% map_from_rowID_to_item(rowID:end) = [];
 
item_user_list(item_user_list(:,1) == 0,:) = [];
item_user_list(item_user_list(:,2) == 0,:) = [];
R = sparse(item_user_list(:,1), item_user_list(:,2), item_user_list(:,3));
mask = sparse(item_user_list(:,1), item_user_list(:,2), ones(size(item_user_list,1),1));
 
time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);
 
 
 
%% reading Item network
 
 
% item1_item2_list = zeros(length(item_network_data),2);
 
tic;
if partial
    fid = fopen(item_network_path, 'r');
end
 
lat_list = zeros(length(map_from_rowID_to_item),1);
lon_list = zeros(length(map_from_rowID_to_item),1);
 
line = 0;
while true
    line = line + 1;
    if partial
        if feof(fid) || line >= reading_line
            break;
        end
        line_str = fgetl(fid);
        line_data = jsondecode(line_str);
    else
        if line >= length(item_network_data)
            break;
        end
        line_str = item_network_data{line};
        line_data = jsondecode(line_str);
    end
    
    
    item = line_data.business_id;
    lat = line_data.latitude;
    lon = line_data.longitude;
    category = line_data.categories;
    
    if isKey(map_from_item_to_rowID, item ) && ~isempty(lat) && ~isempty(lon)
        rowID = map_from_item_to_rowID(item);
        map_from_rowID_to_category{rowID} = category;
        lat_list(rowID) = lat;
        lon_list(rowID) = lon;
    end
end
 
if partial
    fclose(fid);
end
% 
% item1_item2_list(line+1:end,:) = [];
% 
% item_matrix = sparse(item1_item2_list(:,1), item1_item2_list(:,2), ones(size(item1_item2_list,1),1));
 
lat = lat_list;
lon = lon_list;
 
sample_portion = 0.01;
threshold = 0.2;
 
num_spot = length(lat);
num_base = round(num_spot * sample_portion);
 
rng('default')
base_id = datasample(1:num_spot, num_base,'Replace', false);
 
lat_base = lat(base_id);
lon_base = lon(base_id);
 
base = [lat_base, lon_base];
lat_lon = [lat, lon];
distance_eculidean = pdist2(base, lat_lon); 
edge_exist = (distance_eculidean <= threshold);
 
edge = zeros(10000000, 2);
iii = 1;
for i = 1:size(edge_exist,1)
    poi_base = base_id(i);
    num_neighbors = sum(edge_exist(i,:));
    if iii + num_neighbors - 1 > size(edge,1)
        edge = [edge; zeros(size(edge,1),2)];
    end
    edge(iii : iii + num_neighbors - 1,:) = [poi_base * ones(num_neighbors, 1), find(edge_exist(i,:))'];
    iii = iii + num_neighbors;
end
edge(iii:end, :) = [];
edge(edge(:,1) == edge(:,2), :) = [];
item_matrix = sparse(edge(:,1), edge(:,2), ones(size(edge,1),1));
time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);
 
%% Reading social data

tic;
if partial
    fid = fopen(social_network_path, 'r');
end
 
 
line = 0;
edge = zeros(10000000,2);
iii = 1;
while true
    line = line + 1;
    if partial
        if feof(fid) || line >= reading_line
            break;
        end
        line_str = fgetl(fid);
        line_data = jsondecode(line_str);
    else
        if line >= length(social_network_data)
            break;
        end
        line_str = social_network_data{line};
        line_data = jsondecode(line_str);
    end
    
    
    user = line_data.user_id;
    userID = map_from_user_to_colID(user);
    friends = line_data.friends;
    
    
    
    if ~strcmp(friends, 'None')
        friends = split(friends, ', ');
        if (iii + length(friends) - 1) > size(edge,1)
            edge = [edge; zeros(size(edge,1),2)];
        end
        for i = 1:length(friends)
            friend = friends{i};
            try
                friendID = map_from_user_to_colID(friend);
                edge(iii, 1) = userID;
                edge(iii, 2) = friendID;
                iii = iii + 1;
%                 disp('OKOKOKOK');
            catch
%                 friendID = userID;
%                 fprintf('warning: userID[%d] without friend[%s]\n',userID,friend);
            end 
        end
    end
end
edge_1 = edge;
edge(iii,:) = [];
edge(edge(:,1) == edge(:,2), :) = [];
 
 
social_matrix = sparse(edge(:,1), edge(:,2), ones(size(edge,1),1));
 
time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);
 
%% Data saving
 
clear map;
map.map_from_user_to_colID = map_from_user_to_colID;
map.map_from_colID_to_user = map_from_colID_to_user;
map.map_from_item_to_rowID = map_from_item_to_rowID;
map.map_from_rowID_to_item = map_from_rowID_to_item;
map.map_from_rowID_to_category = map_from_rowID_to_category;

%

num_user = length(map.map_from_user_to_colID);
num_item = length(map.map_from_item_to_rowID);

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

save('data/yelp_item188593_user1518169.mat', 'social_matrix', 'R', 'mask','item_matrix', 'map');
fprintf('Data saved successully!\n');

%% Obtain small dataset

user_threshold = 100;
item_threshold = 200;


reserved_user_ID = sum(R > 0,1) >= user_threshold;
reserved_item_ID = sum(R > 0,2) >= item_threshold;

fprintf("reserved user number: [%d]\n", full(sum(reserved_user_ID)));
fprintf("reserved item number: [%d]\n", full(sum(reserved_item_ID)));

remove(map.map_from_item_to_rowID, map.map_from_rowID_to_item(~reserved_item_ID));
remove(map.map_from_user_to_colID, map.map_from_colID_to_user(~reserved_user_ID));
map.map_from_colID_to_user = map.map_from_colID_to_user(reserved_user_ID);
map.map_from_rowID_to_category = map.map_from_rowID_to_category(reserved_item_ID);
map.map_from_rowID_to_item = map.map_from_rowID_to_item(reserved_item_ID);

social_matrix = social_matrix(reserved_user_ID, reserved_user_ID);
item_matrix = item_matrix(reserved_item_ID, reserved_item_ID);
R = R(reserved_item_ID, reserved_user_ID);
mask = mask(reserved_item_ID, reserved_user_ID);

save('data/yelp_small.mat', 'social_matrix', 'R', 'mask','item_matrix', 'map');
fprintf('Small Data saved successully!\n');


