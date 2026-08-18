
class ContextCluster():


    def __init__(self,cluster_threshold):

        self.cluster_threshold = cluster_threshold

    def process_context_for_cluster(self, cur_context):
        if cur_context is None:
            return None, None, None

        if_belong_cluster = 1
        extreme_context = {}
        cluster_name = '' 

        if self.cluster_threshold == 0:
            return None, None, None

        elif ( 'band_Mbps' in cur_context ) and ( 'obj_size_norm' in cur_context ) and \
           ( 'obj_num' in cur_context ) and  ( 'obj_speed' in cur_context):
            
            band_Mbps =  cur_context['band_Mbps']
            obj_size_norm = cur_context['obj_size_norm']
            obj_num = cur_context['obj_num']
            obj_speed = cur_context['obj_speed']

            if (band_Mbps is None) or (obj_size_norm is None) or (obj_num is None) or (obj_speed is None):
                return None, None, None


            if band_Mbps < 0.1:
                cluster_name += '0'
                extreme_context['band_Mbps'] = 0
            elif band_Mbps < 1:
                cluster_name += '1'
                extreme_context['band_Mbps'] = 0.1
                if band_Mbps > 0.1 + (1-0.1)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif band_Mbps < 5:
                cluster_name += '2'
                extreme_context['band_Mbps'] = 1
                if band_Mbps > 1 + (5-1)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif band_Mbps < 10:
                cluster_name += '3'
                extreme_context['band_Mbps'] = 5
                if band_Mbps > 5 + (10-5)*self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '4'
                extreme_context['band_Mbps'] = 10
                if band_Mbps > 10 + 10*self.cluster_threshold:
                    if_belong_cluster = 0

            if obj_size_norm < 0.05:
               cluster_name += '0'
               extreme_context['obj_size_norm'] = 0.01
            elif obj_size_norm < 0.1:
                cluster_name += '1'
                extreme_context['obj_size_norm'] = 0.05
                if obj_size_norm > 0.05 + (0.1-0.05)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_size_norm < 0.2:
                cluster_name += '2'
                extreme_context['obj_size_norm'] = 0.1
                if obj_size_norm > 0.1 + (0.2-0.1)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_size_norm < 0.3:
                cluster_name += '3'
                extreme_context['obj_size_norm'] = 0.2
                if obj_size_norm > 0.2 + (0.3-0.2)*self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '4'
                extreme_context['obj_size_norm'] = 0.3
                if obj_size_norm > 0.3 + 0.3*self.cluster_threshold:
                    if_belong_cluster = 0
            
            if obj_num < 1:
                cluster_name += '0'
                extreme_context['obj_num'] = 1
            elif obj_num < 5:
                cluster_name += '1'
                extreme_context['obj_num'] = 5
                if obj_num < 5 - (5-1)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_num < 10:
                cluster_name += '2'
                extreme_context['obj_num'] = 10
                if obj_num < 10 - (10-5)*self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '3'
                extreme_context['obj_num'] = 20
                if obj_num < 20 - (20-10)*self.cluster_threshold or obj_num > 20:
                    if_belong_cluster = 0
            

            if obj_speed < 260:
                cluster_name += '0'
                extreme_context['obj_speed'] = 260
            elif obj_speed < 520:
                cluster_name += '1'
                extreme_context['obj_speed'] = 520
                if obj_speed < 520 - (520-260)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_speed < 780:
                cluster_name += '2'
                extreme_context['obj_speed'] = 780
                if obj_speed < 780 - (780-520)*self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '3'
                extreme_context['obj_speed'] = 1500
                if obj_speed < 1500 - (1500-780)*self.cluster_threshold or obj_speed > 1500:
                    if_belong_cluster = 0
            
            return cluster_name, extreme_context, if_belong_cluster
        
        else:
            return None, None, None


