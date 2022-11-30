""" Architect controls architecture of cell by computing gradients of alphas 
    NAS训练算法中的第1步: 更新架构参数 α
    根据论文可知 dα Lval(w*, α) 约等于 dα Lval(w', α)    w' = w - ξ * dw Ltrain(w, α)
"""
import copy
import torch


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net                      # network
        self.v_net = copy.deepcopy(net)     # 不直接用外面的optimizer来进行w的更新，而是自己新建一个network，主要是因为我们这里的更新不能对Network的w进行更新
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay    # 正则化项用来防止过拟合

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        根据公式计算 w' = w - ξ * dw Ltrain(w, α)   
        Monmentum公式：  dw Ltrain -> v * w_momentum + dw Ltrain + w_weight_decay * w 
        -> m + g + 正则项
  
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)  即公式中的 ξ
            w_optim: weights optimizer 用来更新 w 的优化器
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)

        # compute gradient 计算  dw L_trn(w) = g
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                # m = v * w_momentum  用的就是Network进行w更新的momentum
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum

                # 做一步momentum梯度下降后更新得到 w' = w - ξ * (m + dw Ltrain(w, α) + 正则项 )
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))   

            # synchronize alphas 更新了v_net的alpha
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    # main入口=====================================
    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        计算目标函数关于 α 的近似梯度
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y) # L_val(w', α)  在使用w', 新alpha的net上计算损失值

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]    # dα L_val(w', α)   梯度近似后公式第一项
        dw = v_grads[len(v_alphas):]        # dw' L_val(w', α)  梯度近似后公式第二项的第二个乘数

        hessian = self.compute_hessian(dw, trn_X, trn_y)        # 梯度近似后公式第二项

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h    # 求出了目标函数的近似梯度值

    # 被unrolled_backward调用
    def compute_hessian(self, dw, trn_X, trn_y):   
        """
        求经过泰勒展开后的第二项的近似值
        dw = dw` { L_val(w`, alpha) }  输入里已经给了所有预测数据的dw
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)    [1]
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()   # 把每个 w 先拉成一行，然后把所有的 w 摞起来，变成 n 行, 然后求L2值
        eps = 0.01 / norm

        # w+ = w + eps * dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d        # 将model中所有的w'更新成 w+
        loss = self.net.loss(trn_X, trn_y)      # L_trn(w+)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps * dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d   # 将model中所有的w'更新成 w-,   w- = w - eps * dw = w+ - eps * dw * 2, 现在的 p 是 w+
        loss = self.net.loss(trn_X, trn_y)      # L_trn(w-)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d        # 将模型的参数从 w- 恢复成 w,  w = w- + eps * dw

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]  # 利用公式 [1] 计算泰勒展开后第二项的近似值返回
        return hessian
