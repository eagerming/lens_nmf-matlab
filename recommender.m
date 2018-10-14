function rating =  recommender(V, U, ratingIDpair)

R = V * U;

if size(ratingIDpair,2) == 1
    rating = R(:,ratingIDpair);
    return;
end

num = size(ratingIDpair,1);
rating = zeros(num,1);

for i = 1 : num
    rating(i) = R(ratingIDpair(i,1) , ratingIDpair(i,1));
end
