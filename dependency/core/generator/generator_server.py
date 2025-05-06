from core.lib.common import Context


class GeneratorServer:

    @staticmethod
    def run():
        source_id = Context.get_parameter('SOURCE_ID', direct=False)
        source_type = Context.get_parameter('SOURCE_TYPE')
        source_url = Context.get_parameter('SOURCE_URL')
        source_metadata = Context.get_parameter('SOURCE_METADATA', direct=False)
        dag = Context.get_parameter('DAG', direct=False)

        generator = Context.get_algorithm('GENERATOR', al_name=source_type,
                                          source_id=source_id, source_url=source_url,
                                          source_metadata=source_metadata, dag=dag)
        generator.run()
