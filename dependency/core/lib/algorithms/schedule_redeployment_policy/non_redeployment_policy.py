from .base_redeployment_policy import BaseRedeploymentPolicy
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('NonRedeploymentPolicy',)

@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='non')
class NonRedeploymentPolicy(BaseRedeploymentPolicy):
    """No-operation redeployment policy"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # 记录初始化信息（可选）
        LOGGER.debug('[Redeployment] NON policy initialized')
    
    def __call__(self, info):
        """Return empty redeployment plan
        
        Args:
            info (dict): Context information containing:
                - source: Information about the data source
                - dag: Service dependency graph
                - node_set: Available node set
        
        Returns:
            dict: Always empty dictionary {}
        """
        source_id = info['source']['id']
        # 明确记录不执行重新部署
        LOGGER.info(f'[Redeployment] Source {source_id}: NO redeployment (non policy)')
        
        # 返回空字典表示无操作
        return {}