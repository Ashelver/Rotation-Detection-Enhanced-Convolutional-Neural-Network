from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class Single_Oscillate:
    def __init__(self, image_tensor):
        self.block_size = 4
        self.image = image_tensor[0]
        self.new_image_one = torch.zeros(28, 28)
        self.new_image_two = torch.zeros(28, 28)
        self.grid_size = 7
        # Ensure the image size is 28x28 (resize if necessary)
        if self.image.shape != (28, 28):
            raise Warning("This method is designed for 28x28 images")
        self.blocks = []
        # Split the image into patches
        for i in range(self.grid_size):  # 7x7 grid
            for j in range(self.grid_size):
                block = self.image[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
                self.blocks.append(block)

    def get_ring_orders_one(self, row, column):
        if row < column and row + column <= self.grid_size - 1:
            # Right
            return self.blocks[row*self.grid_size + column-1]
        if row >= column and row + column < self.grid_size - 1:
            # Up
            return self.blocks[(row+1)*self.grid_size + column]
        if row > column and row + column >= self.grid_size - 1:
            # Left
            return self.blocks[row*self.grid_size + column+1]
        if row <= column and row + column > self.grid_size - 1:
            # Down
            return self.blocks[(row-1)*self.grid_size + column]
        return self.blocks[row*self.grid_size + column]


    def get_ring_orders_two(self, row, column):
        if row <= column and row + column < self.grid_size - 1:
            # Left
            return self.blocks[row*self.grid_size + column+1]
        if row > column and row + column <= self.grid_size - 1:
            # Down
            return self.blocks[(row-1)*self.grid_size + column]
        if row >= column and row + column > self.grid_size - 1:
            # Right
            return self.blocks[row*self.grid_size + column-1]            
        if row < column and row + column >= self.grid_size - 1:
            # Up
            return self.blocks[(row+1)*self.grid_size + column]
        return self.blocks[row*self.grid_size + column]

    def add_value(self, row, column):
        self.new_image_one[(row*self.block_size):((row+1)*self.block_size), (column*self.block_size):((column+1)*self.block_size)] += self.get_ring_orders_one(row, column)
        self.new_image_two[(row*self.block_size):((row+1)*self.block_size), (column*self.block_size):((column+1)*self.block_size)] += self.get_ring_orders_two(row, column)


    def get_result(self):
        for i in range(self.grid_size):  # 7x7 grid
            for j in range(self.grid_size):
                self.add_value(i,j)
        mean_kernel = torch.ones(1, 1, 3, 3) / 9.0
        # return F.conv2d(((0.4*self.image + 0.3*self.new_image_one + 0.3*self.new_image_two).unsqueeze(0)), mean_kernel, padding=1, groups=1)
        return F.conv2d((self.new_image_one.unsqueeze(0)), mean_kernel, padding=1, groups=1)
    


class BatchWise_Oscillate:
    def __init__(self, image_tensor):
        self.block_size = 4
        self.images = image_tensor
        self.batch_size = image_tensor.size(0)  # Get the batch size
        self.new_images_one = torch.zeros_like(self.images)
        self.new_images_two = torch.zeros_like(self.images)
        self.grid_size = 7
        # Ensure the image size is 28x28 (resize if necessary)
        if self.images[0].shape != (1, 28, 28):
            raise Warning("This method is designed for 28x28 images")
        self.blocks = []
        # Split the image into patches
        for i in range(self.grid_size):  # 7x7 grid
            for j in range(self.grid_size):
                block = self.images[:,0,i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
                block = self.images[:, 0, i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
                self.blocks.append(block)

    def get_ring_orders_one(self, row, column):
        if row < column and row + column <= self.grid_size - 1:
            # Right
            return self.blocks[row*self.grid_size + column-1]
        if row >= column and row + column < self.grid_size - 1:
            # Up
            return self.blocks[(row+1)*self.grid_size + column]
        if row > column and row + column >= self.grid_size - 1:
            # Left
            return self.blocks[row*self.grid_size + column+1]
        if row <= column and row + column > self.grid_size - 1:
            # Down
            return self.blocks[(row-1)*self.grid_size + column]
        return self.blocks[row*self.grid_size + column]


    def get_ring_orders_two(self, row, column):
        if row <= column and row + column < self.grid_size - 1:
            # Left
            return self.blocks[row*self.grid_size + column+1]
        if row > column and row + column <= self.grid_size - 1:
            # Down
            return self.blocks[(row-1)*self.grid_size + column]
        if row >= column and row + column > self.grid_size - 1:
            # Right
            return self.blocks[row*self.grid_size + column-1]            
        if row < column and row + column >= self.grid_size - 1:
            # Up
            return self.blocks[(row+1)*self.grid_size + column]
        return self.blocks[row*self.grid_size + column]


    def add_value(self, row, column):
        self.new_images_one[:, 0,(row*self.block_size):((row+1)*self.block_size), (column*self.block_size):((column+1)*self.block_size)] += self.get_ring_orders_one(row, column)
        self.new_images_two[:, 0,(row*self.block_size):((row+1)*self.block_size), (column*self.block_size):((column+1)*self.block_size)] += self.get_ring_orders_two(row, column)

    def get_result(self):
        for i in range(self.grid_size):  # 7x7 grid
            for j in range(self.grid_size):
                self.add_value(i,j)
        mean_kernel = torch.ones(1, 1, 3, 3) / 9.0
        # return 0.4*self.images + 0.3*self.new_images_one + 0.3*self.new_images_two
        return self.new_images_one, self.new_images_two
    



if __name__ == "__main__":
    image_path = "../imgs/output_image.png"
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    print(image_tensor.shape)
    image_test = torch.rand(64, 1, 28, 28)
    # plt.imshow(transforms.ToPILImage()(Single_Oscillate(image_tensor).get_result()), cmap='gray')
    # print(Single_Oscillate(image_tensor).get_result().shape)
    plt.imshow(transforms.ToPILImage()(BatchWise_Oscillate(image_test).get_result()[0][0]), cmap='gray')
    plt.title("New Image after Oscillation")
    plt.axis('off')
    plt.show()





# from PIL import Image
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms

# # ================================
# # (0,0): right 4
# # (0,1): right 3
# # (0,2): right 2, down 2
# # (0,3): down 4
# # (1,0): up 2, right 2
# # (1,1): right 2
# # (1,2): down 2
# # (1,3): down 3
# # (2,0): up 3
# # (2,1): up 2
# # (2,2): left 2
# # (2,3): left 2, down 2
# # (3,0): up 4
# # (3,1): left 2, up 2
# # (3,2): left 3
# # (3,3): left 4
# # ================================

# class Single_Oscillate:
#     def __init__(self, image_tensor):
#         self.block_size = 7
#         self.image = image_tensor[0]
#         self.new_image = torch.zeros(28, 28)
#         # Ensure the image size is 28x28 (resize if necessary)
#         if self.image.shape != (28, 28):
#             raise Warning("This method is designed for 28x28 images")
#         self.blocks = []
#         # Split the image into patches
#         for i in range(4):  # 4x4 grid
#             for j in range(4):
#                 block = self.image[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
#                 self.blocks.append(block)

#     def get_blocks(self, row, column):
#         return self.blocks[row*4 + column]

#     def add_value(self, row, column, hor, ver):
#         self.new_image[(row*self.block_size+hor):((row+1)*self.block_size+hor), (column*self.block_size+ver):((column+1)*self.block_size+ver)] += self.get_blocks(row, column)

#     def get_result(self):
#         self.add_value(0,0,4,0)
#         self.add_value(0,1,3,0)
#         self.add_value(0,2,2,-2)
#         self.add_value(0,3,0,-4)
#         self.add_value(1,0,2,2)
#         self.add_value(1,1,2,0)
#         self.add_value(1,2,0,-2)
#         self.add_value(1,3,0,-3)
#         self.add_value(2,0,0,3)
#         self.add_value(2,1,0,2)
#         self.add_value(2,2,-2,0)
#         self.add_value(2,3,-2,-2)
#         self.add_value(3,0,0,4)
#         self.add_value(3,1,-2,2)
#         self.add_value(3,2,-3,0)
#         self.add_value(3,3,-4,0)
#         return (self.new_image*0.3 + self.image*0.4 + torch.flip(self.new_image,dims=[1])*0.3).unsqueeze(0)


# class BatchWise_Oscillate:
#     def __init__(self, image_tensor):
#         self.block_size = 7
#         self.images = image_tensor  # image_tensor is expected to have shape [batch_size, 1, 28, 28]
#         self.batch_size = image_tensor.size(0)  # Get the batch size
#         self.new_images = torch.zeros_like(self.images)  # Initialize a tensor to store processed images
#         # Ensure the image size is 28x28 (resize if necessary)
#         if self.images[0].shape != (1, 28, 28):
#             raise Warning("This method is designed for batchsize x 1 x 28x28 tensor")
#         self.blocks = []
#         # Split the image into patches
#         for i in range(4):  # 4x4 grid
#             for j in range(4):
#                 block = self.images[:,0,i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
#                 self.blocks.append(block)

#     def get_blocks(self, row, column):
#         return self.blocks[row*4 + column]

#     def add_value(self, row, column, hor, ver):
#         self.new_images[:, 0,(row*self.block_size+hor):((row+1)*self.block_size+hor), (column*self.block_size+ver):((column+1)*self.block_size+ver)] += self.get_blocks(row, column)

#     def get_result(self):
#         self.add_value(0,0,4,0)
#         self.add_value(0,1,3,0)
#         self.add_value(0,2,2,-2)
#         self.add_value(0,3,0,-4)
#         self.add_value(1,0,2,2)
#         self.add_value(1,1,2,0)
#         self.add_value(1,2,0,-2)
#         self.add_value(1,3,0,-3)
#         self.add_value(2,0,0,3)
#         self.add_value(2,1,0,2)
#         self.add_value(2,2,-2,0)
#         self.add_value(2,3,-2,-2)
#         self.add_value(3,0,0,4)
#         self.add_value(3,1,-2,2)
#         self.add_value(3,2,-3,0)
#         self.add_value(3,3,-4,0)
#         # return 
#         return self.new_images


# if __name__ == "__main__":
#     image_path = "output_image.png"
#     image = Image.open(image_path)
#     transform = transforms.ToTensor()
#     image_tensor = transform(image)
#     image_test = torch.ones(64, 1, 28, 28)
#     sample = BatchWise_Oscillate(image_test)
#     plt.imshow(transforms.ToPILImage()(sample.get_result()[0,0]), cmap='gray')
#     print(sample.get_result()[0,0])
#     plt.title("New Image after Oscillation")
#     plt.axis('off')
#     plt.show()
