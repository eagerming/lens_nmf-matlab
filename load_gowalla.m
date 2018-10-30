% function load_gowalla()
%%
% fid = fopen('data/gowalla/visited_spots.txt','r');
gowalla_data = importdata('data/gowalla/visited_spots.txt');
reading_line = 500;
num_of_line = length(gowalla_data);


R = sparse(1);
for i = 1:min(reading_line, num_of_line)
    tline = gowalla_data{i};
    user_id = extractBefore(tline, ' [');
    poi_id = extractAfter(tline, '[');
    poi_id = extractBefore(poi_id,  ']');
    poi_list = split(poi_id, ', ');
    poi_list = str2num(char(poi_list));
    R(poi_list, user_id) = 1;
end


% fclose(fid);

%%

fid = fopen('data/gowalla/spot_location.txt','r');
reading_line = 500;
C = textscan(fid, '%d %f %f\n', reading_line);
fclose(fid);

poi = C{1};
lon = C{2};
lat = C{3};

lat_max = max(lat);
lat_min = min(lat);
lon_max = max(lon);
lon_min = min(lon);


sample_portion = 0.01;
distance_threshold = 10000;

num_spot = length(poi);
num_base = round(num_spot * sample_portion);

rng('default')
base_id = datasample(1:num_spot, num_base,'Replace', false);

lat_base = lat(base_id);
lon_base = lon(base_id);

item_matrix = sparse(1);
for i = 1:num_base
    neighbor = [];
    poi_base = poi(base_id(i));
    for j = 1:num_spot
        distt = Dist(lat_base(i), lon_base(i), lat(j), lon(j));
        if distt <= distance_threshold
            neighbor = [neighbor poi(j)];
        end
    end
    item_matrix(poi_base, neighbor) = 1;
	item_matrix(neighbor, poi_base) = 1;
end

% end

%%
function distt = Dist(lat1, lon1, lat2, lon2)
%     lon1 = 104.07075;
%     lat1 = 30.664283;
%     lon2 = 103.93507;
%     lat2 = 30.760159;
    
    distt = distance(lat1,lon1,lat2,lon2);
    distt = distt*6371*1000*2*pi/360;
end