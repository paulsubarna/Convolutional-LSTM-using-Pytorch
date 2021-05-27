import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, keep_dims= None, peephole= None,device=None):
        super(ConvLSTMCell, self).__init__()
        
        
        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.peephole = peephole
        self.device=device
        self.padding = int((kernel_size - 1) // 2)
        self.odd_kernel= None
        self.keep_dims= keep_dims
        
        # Findout whether kernel is even or odd to define the desired size of hidden_layers and peephole connections 
        
        if kernel_size % 2 == 0:
            self.odd_kernel= False
        else:
            self.odd_kernel= True
        
        if self.keep_dims:
            
            #self.Wx = nn.Conv1d(self.input_channels+self.hidden_channels, self.hidden_channels * 4, self.kernel_size, stride=1, padding=self.padding, bias=True)
            self.Wx = nn.Conv1d(self.input_channels, self.hidden_channels * 4, self.kernel_size, stride=1, padding=self.padding,bias=True)
            self.Wh = nn.Conv1d(self.hidden_channels, self.hidden_channels * 4, self.kernel_size, stride=1, padding=self.padding, bias=True)
        
        else:
            #self.Wx = nn.Conv1d(self.input_channels+ self.hidden_channels, self.hidden_channels * 4, self.kernel_size, stride=1,  bias=True)
            self.Wx = nn.Conv1d(self.input_channels, self.hidden_channels * 4, self.kernel_size, stride=1, bias=True)
            self.Wh = nn.Conv1d(self.hidden_channels, self.hidden_channels * 4, self.kernel_size, stride=1,  bias=True)
            
        self.Wci = None
        self.Wcf = None
        self.Wco = None
        

    def forward(self, x, h, c):
        
        if self.keep_dims:
            print('Dimension are same')
            conv_x = self.Wx(x)
            print(conv_x.shape)
            conv_h = self.Wh(h)
            print(conv_h.shape)
            ci_x, cf_x, cc_x, co_x = torch.split( conv_x, self.hidden_channels, dim=1)

            ci_h, cf_h, cc_h, co_h = torch.split( conv_h, self.hidden_channels, dim=1)
            #print(ci_h.shape)
        else:
            print('Dimensions are not same')
            conv_x = self.Wx(x)
            print('the shape of conv_x:',conv_x.shape)
            conv_h = self.Wh(h)
            print('the shape of conv_x:',conv_h.shape)
            ci_x, cf_x, cc_x, co_x = torch.split( conv_x, self.hidden_channels, dim=1)

            ci_h, cf_h, cc_h, co_h = torch.split( conv_h, self.hidden_channels, dim=1)
            #print('the shape of ci_h:',ci_h.shape)
        
        if self.peephole:
            
            bs,_,seq_c= self.test_convolution(x).size()
            
            self.Wci = nn.Parameter(torch.zeros(1, self.hidden_channels,seq_c)).to(self.device)
            self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_channels,seq_c)).to(self.device)
            self.Wco = nn.Parameter(torch.zeros(1, self.hidden_channels,seq_c)).to(self.device)

            ci = torch.sigmoid(ci_x + ci_h + c * self.Wci)
            cf = torch.sigmoid(cf_x + cf_h  + c * self.Wcf)
            cc = cf * c + ci * torch.tanh(cc_x + cc_h )
            co = torch.sigmoid( co_x + co_h + cc * self.Wco)
            
            
        else:
                       
            ci = torch.sigmoid(ci_x + ci_h )
            cf = torch.sigmoid(cf_x + cf_h )
            cc = cf * c + ci * torch.tanh(cc_x + cc_h )
            co = torch.sigmoid( co_x + co_h)
        
        
        ch = co * torch.tanh(cc)
        return ch, cc

    def test_convolution(self,x):
        return self.Wx(x)
        
    
    def init_hidden(self, batch_size, hidden, x,device=None):
        
        batch_size,_,seq_h=x.size()
        print(seq_h)
        bs,_,seq_c= self.test_convolution(x).size()
        if self.odd_kernel:
            if self.peephole:
                self.Wci = nn.Parameter(torch.zeros(1, hidden,seq_c)).to(device)
                self.Wcf = nn.Parameter(torch.zeros(1, hidden,seq_c)).to(device)
                self.Wco = nn.Parameter(torch.zeros(1, hidden,seq_c)).to(device)

            state= (Variable(torch.zeros(batch_size, hidden, seq_h)).to(device),
                Variable(torch.zeros(batch_size, hidden, seq_c)).to(device))
        else:
            
            if self.peephole:
                self.Wci = nn.Parameter(torch.zeros(1, hidden,seq_c)).to(device)
                self.Wcf = nn.Parameter(torch.zeros(1, hidden,seq_c)).to(device)
                self.Wco = nn.Parameter(torch.zeros(1, hidden,seq_c)).to(device)

            state= (Variable(torch.zeros(batch_size, hidden, seq_h)).to(device),
                Variable(torch.zeros(batch_size, hidden, seq_c)).to(device))
            
        return state


class ConvLSTM(nn.Module):
    # This implementation is intented for vibrational sensor data or time-series data with single dimension or multiple dimension
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    # Expected input shape: (batch, input_features, sequence length)
    #"""""" Parameters ---
    #input_channels: int
    #    no of features or dimensions
    #hidden_channels: list
    #    no of hidden channels
    #kernel_size: list----kernel size for each layer 
    #step: int---no of steps for lstm
    #effective step: list-- effective steps to append into the final output
    #device: string---type of device to work with


      
    
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], keep_dims=None, device=None,peephole=None ):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.keep_dims= keep_dims
        self.peephole = peephole
        self.step = step
        self.device= device
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            if i==0:
                self.input_channels = self.input_channels
            else:
                self.input_channels= self.hidden_channels[i-1]
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels, self.hidden_channels[i], self.kernel_size[i],self.keep_dims,self.peephole,self.device)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        cell_state = []
        outputs = []
        x=input
          
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            bsize, _, seq = x.size()
            #if i==0:       
            (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],x=x, device=self.device)
            cell_state.append((h, c))
            #(h, c) = cell_state[i]
            
            if i != 0:
                if not self.keep_dims:
                    bsize,_,seq= (getattr(self,name).test_convolution(x=h)).size()
                    c=c[:,:,:seq]
                elif self.keep_dims and self.kernel_size[i]%2==0 :
                    bsize,_,seq= (getattr(self,name).test_convolution(x=h)).size()
                    c=c[:,:,:seq] 
                
            print('Shape of h is {} & c is {} of layer {}'.format( h.shape, c.shape,i))
            
            hidden_curr, c_curr = getattr(self, name)(x, h, c)
            cell_state.append((hidden_curr, c_curr))
            outputs.append(hidden_curr)
            x=hidden_curr
            
        return outputs, (hidden_curr, c_curr)

