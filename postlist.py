import math

# Node class

def encode_varbyte(n):
    bytes_needed = math.ceil(n.bit_length() / 7)
    result = bytearray()
    for i in range(bytes_needed):
        byte = n & 0x7F
        n >>= 7
        if i != bytes_needed - 1:
            byte |= 0x80
        result.append(byte)
    return bytes(result)

def decode_varbyte(encoded):
    value = 0
    shift = 0
    for byte in encoded:
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            break
        shift += 7
    return value

class Node:
    # Function to initialize the node object
    def __init__(self, key, position):
        self.data = key
        self.tf = 1
        self.positions = [position]
        self.next = None
        self.skip = None
        self.currentPosition = position
        
    def addPosition(self, position):
        self.positions.append(encode_varbyte(position)) #does not check if position already exists (save time)
    
    # Function to view node
    def __str__(self):
        res = "({}, {}): {}".format(self.data, self.tf, self.positions)
        return res

    def __repr__(self):
        return self.__str__()
 
# Linked List class
class PostingList:
    # Function to initialize the Posting List object 
    # (based on Linked List with skip pointers)
    def __init__(self):
        self.head = None
        self.tail = None
        self.df = 0

    # Function to view list
    def __str__(self):
        res = []
        curr = self.head
        while curr is not None:
            res.append(str(curr))
            curr = curr.next
        return '; '.join(map(str, res))

    def __repr__(self):
        return self.__str__()

    # insert to the end of the list since document id is already in sorted order
    def insertNode(self, data, position):
        if self.head is None:
            n = Node(data, encode_varbyte(position))
            # n.data = data
            # n.addPosition(position)
            self.head = n
            self.tail = n
            self.df += 1
            return
        
        curr = self.tail
        
        if curr.data == data:
            last_seen_positional_index_byte = curr.currentPosition
            last_seen_positional_index_int = decode_varbyte(last_seen_positional_index_byte)
            offset = position - last_seen_positional_index_int
            curr.addPosition(offset)
            curr.currentPosition = encode_varbyte(position)
            curr.tf += 1
            return
        
        if curr.data < data:
            n = Node(data, encode_varbyte(position))
            curr.next = n
            self.tail = n
            self.df += 1
            return

    # Add evenly spaced pointers to posting list   
    def addSkipPointer(self):
        skip = math.floor(math.sqrt(self.df))
        n = 0
        curr = self.head
        skip_temp = curr

        while (n + skip < self.df):
            initial_skip_temp = skip_temp
            for i in range(skip):
                skip_temp = skip_temp.next
            initial_skip_temp.skip = skip_temp
            n += skip
        return

"""
test = Node(5, 1)
#print out --> (docID, termFreq): [positional indices]
--> (5, 1): [1] 

test = PostingList()
test.addNode(5, 1)
test.addNode(5, 4)
test.addNode(7, 1)
test.addNode(7, 2)
test.addNode(7, 3)
test.addNode(8, 100)
#print out --> (docID_1, termFreq): [positional indices]; (docID_2, termFreq): [positional indices]; ...
--> (5, 2): [1, 4]; (7, 3): [1, 2, 3]; (8, 1): [1, 100]
"""

"""
>>> from postlist import *
>>> pl = PostingList()
>>> pl.insertNode(1,2)
(1, 1): [2]
>>> pl.insertNode(2,2)
>>> pl.insertNode(3,2)
>>> pl.insertNode(4,2)
>>> pl.insertNode(5,2)
>>> pl.insertNode(6,2)
>>> pl.insertNode(6,3)
>>> pl
(1, 1): [2]; (2, 1): [2]; (3, 1): [2]; (4, 1): [2]; (5, 1): [2]; (6, 2): [2, 3]
"""