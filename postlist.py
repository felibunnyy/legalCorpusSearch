import math

# Node class
class Node:
    # Function to initialize the node object
    def __init__(self):
        self.data = None 
        self.next = None
        self.skip = None
    
    def getHead(self):
        return self.head
    
    def getSkip(self):
        return self.skip
    
    # Function to view node
    def __str__(self):
        currNode = self.data
        nextNode = self.next.data if self.next is not None else None
        skipNode = self.skip.data if self.skip is not None else None
        data = [currNode, nextNode, skipNode]
        return "%s" %(' -> '.join(str(i) for i in data))

    def __repr__(self):
        return self.__str__()
 
# Linked List class
class PostingList:
    # Function to initialize the Posting List object 
    # (based on Linked List with skip pointers)
    def __init__(self):
        self.head = None
        self.length = 0

    # Function to view list
    def __str__(self):
        data = []
        curr = self.head
        while curr is not None:
            data.append(curr.data)
            curr = curr.next
        return "[%s]" %(', '.join(str(i) for i in data))

    def __repr__(self):
        return self.__str__()

    # Add new docID
    def addNode(self, data):
        curr = self.head
        # Empty list / reach end of list, create new Node
        if curr is None:
            n = Node()
            n.data = data
            self.head = n
            self.length += 1
            return

        # input docID is smaller than current pointer, insert before current
        if curr.data > data:
            n = Node()
            n.data = data
            n.next = curr
            self.head = n
            self.length += 1
            return

        if curr.data == data:
            return

        # input docID is larger than current pointer, move forward
        while curr.next is not None:
            if curr.next.data == data:
                return
            if curr.next.data > data:
                break
            curr = curr.next
        n = Node()
        n.data = data
        n.next = curr.next
        curr.next = n
        self.length += 1
        return

    # Assign evenly spaced pointers to each node in the list
    def addSkipPointer(self):
        skip = math.floor(math.sqrt(self.length))
        n = 0
        curr = self.head
        while (n < self.length) & (n + skip < self.length):
            skip_temp = curr
            for i in range(skip): 
                skip_temp = skip_temp.next
            curr.skip = skip_temp
            curr = curr.skip
            n += skip
        return

def mergeLists(headA, headB):
 
    # A dummy node to store the result
    dummyNode = Node(0)
 
    # Tail stores the last node
    tail = dummyNode
    while True:
 
        # If any of the list gets completely empty
        # directly join all the elements of the other list
        if headA is None:
            tail.next = headB
            break
        if headB is None:
            tail.next = headA
            break
 
        # Compare the data of the lists and whichever is smaller is
        # appended to the last of the merged list and the head is changed
        if headA.data <= headB.data:
            tail.next = headA
            headA = headA.next
        else:
            tail.next = headB
            headB = headB.next
 
        # Advance the tail
        tail = tail.next
 
    # Returns the head of the merged list
    return dummyNode.next