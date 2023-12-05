function [nearPatch,nearPatch1] = givePatch(a,inputTexture,temp1,mask)
    temp2 = temp1.*temp1.*mask;
    temp3 = inputTexture.*inputTexture;
    temp3 = filter2(mask,temp3,'valid');
    temp4 = filter2(temp1.*mask,inputTexture,'valid');
    errors = sum(temp2(:))+temp3-2*temp4;
    minerror = abs(min(errors(:)));
    [x,y] = find(errors <= minerror*1.3);
    randint = randi([1 length(x)],1);
    [m,n] = size(mask);
    nearPatch = inputTexture(x(randint):x(randint)+m-1,y(randint):y(randint)+n-1);
    nearPatch1 = a(x(randint):x(randint)+m-1,y(randint):y(randint)+n-1,:);
end