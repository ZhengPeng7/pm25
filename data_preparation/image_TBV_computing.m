

%%%%  ����һ���Ҷ�ͼ�񣬼����� TBV ֵ
%%%%  
%%%%
%%%%


function image_TBV = image_TBV_computing(image_gray)

 
[high,width]=size(image_gray);
image_gray=double(image_gray);
image_TBV = 0;

     for i=2:high
         for j=2:width
             image_TBV = image_TBV+(abs(image_gray(i,j)-image_gray(i-1,j))+abs(image_gray(i,j)-image_gray(i,j-1)));
         end
     end

 %%% ���� image_TBV   
end